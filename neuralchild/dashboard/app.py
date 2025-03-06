"""
Dashboard App

This module implements the visualization and interaction dashboard for the NeuralChild system.
It uses Dash to create an interactive web interface for monitoring the child's development
and interacting with the child.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import dash
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from ..core.development import Development
from ..core.child import Child
from ..core.mother import Mother
from ..utils.data_types import (
    DevelopmentalStage, MotherPersonality, DevelopmentConfig,
    ChildResponse, MotherResponse, Emotion, EmotionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Dashboard configuration
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))

# Development configuration
TIME_ACCELERATION = int(os.getenv("TIME_ACCELERATION_FACTOR", "100"))
DEVELOPMENT_SEED = os.getenv("RANDOM_SEED")
if DEVELOPMENT_SEED:
    DEVELOPMENT_SEED = int(DEVELOPMENT_SEED)

# Storage paths
STATES_PATH = os.getenv("STATES_PATH", "./data/states")
os.makedirs(STATES_PATH, exist_ok=True)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="NeuralChild Dashboard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server  # For WSGI deployment

# Global state
development_system = None
conversation_history = []


def initialize_system():
    """Initialize the NeuralChild development system."""
    global development_system
    
    # Check for existing state files
    state_files = [f for f in os.listdir(STATES_PATH) if f.endswith(".json")]
    
    if state_files:
        # Load most recent state
        state_files.sort(key=lambda f: os.path.getmtime(os.path.join(STATES_PATH, f)), reverse=True)
        most_recent = os.path.join(STATES_PATH, state_files[0])
        
        logger.info(f"Loading system state from {most_recent}")
        development_system = Development.load_system_state(most_recent)
    else:
        # Create new system
        logger.info("Creating new development system")
        
        # Create config
        config = DevelopmentConfig(
            time_acceleration_factor=TIME_ACCELERATION,
            random_seed=DEVELOPMENT_SEED,
            mother_personality=MotherPersonality.BALANCED,
            start_age_months=0,
            enable_random_factors=True
        )
        
        # Create child and mother
        child = Child()
        mother = Mother()
        
        # Create development system
        development_system = Development(
            child=child,
            mother=mother,
            config=config
        )


# Create the layout

def create_header():
    """Create the dashboard header."""
    return dbc.Row([
        dbc.Col([
            html.H1("🧠 NeuralChild Development Dashboard", className="dashboard-title"),
            html.P("Monitoring and interacting with a developing neural mind", className="lead")
        ], width=8),
        dbc.Col([
            html.Div([
                html.Button("Save State", id="save-button", className="btn btn-primary me-2"),
                html.Button("Accelerate Development", id="accelerate-button", className="btn btn-warning")
            ], className="d-flex justify-content-end")
        ], width=4)
    ], className="header-row mb-4")


def create_status_cards():
    """Create status cards showing key metrics."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Developmental Stage", className="card-title"),
                    html.H3(id="stage-display", className="stage-value"),
                    html.P(id="age-display", className="text-muted")
                ])
            ], className="status-card")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Key Metrics", className="card-title"),
                    html.Div([
                        html.Div([
                            html.Span("Vocabulary Size: "),
                            html.Span(id="vocabulary-size", className="metric-value")
                        ], className="metric-item"),
                        html.Div([
                            html.Span("Grammatical Complexity: "),
                            html.Span(id="grammatical-complexity", className="metric-value")
                        ], className="metric-item"),
                        html.Div([
                            html.Span("Emotional Regulation: "),
                            html.Span(id="emotional-regulation", className="metric-value")
                        ], className="metric-item")
                    ])
                ])
            ], className="status-card")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Emotional State", className="card-title"),
                    html.Div(id="emotion-indicators", className="emotion-container")
                ])
            ], className="status-card")
        ], width=4)
    ], className="mb-4")


def create_development_chart():
    """Create the developmental metrics chart."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Progress"),
                dbc.CardBody([
                    dcc.Graph(id="development-chart", config={'displayModeBar': False})
                ])
            ], className="chart-card")
        ], width=12)
    ], className="mb-4")


def create_interaction_panel():
    """Create the interaction panel."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Interaction"),
                dbc.CardBody([
                    html.Div(id="conversation-container", className="conversation-container"),
                    dbc.InputGroup([
                        dbc.Input(id="user-input", placeholder="Type something to the child or mother...", type="text"),
                        dbc.Button("Send", id="send-button", color="primary")
                    ], className="mt-3")
                ])
            ], className="interaction-card")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Interaction Controls"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Interaction Mode", className="control-label"),
                        dcc.RadioItems(
                            id="interaction-mode",
                            options=[
                                {"label": "User → Mother → Child", "value": "user_mother_child"},
                                {"label": "User → Child", "value": "user_child"},
                                {"label": "Auto Interaction", "value": "auto"}
                            ],
                            value="user_mother_child",
                            className="mode-selector"
                        )
                    ], className="control-group"),
                    html.Div([
                        html.Label("Mother Personality", className="control-label"),
                        dcc.Dropdown(
                            id="mother-personality",
                            options=[
                                {"label": "Balanced", "value": "balanced"},
                                {"label": "Nurturing", "value": "nurturing"},
                                {"label": "Authoritarian", "value": "authoritarian"},
                                {"label": "Permissive", "value": "permissive"},
                                {"label": "Neglectful", "value": "neglectful"}
                            ],
                            value="balanced",
                            className="personality-selector"
                        )
                    ], className="control-group mt-3"),
                    html.Div([
                        html.Label("Auto Interaction Settings", className="control-label"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Interactions"),
                            dbc.Input(id="auto-count", type="number", value=5, min=1, max=50),
                            dbc.Button("Start", id="auto-start-button", color="secondary")
                        ])
                    ], className="control-group mt-3"),
                    html.Hr(),
                    html.Div([
                        html.H6("Development Acceleration", className="mb-2"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Months"),
                            dbc.Input(id="acceleration-months", type="number", value=6, min=1, max=24),
                        ], className="mb-2"),
                        html.Div(id="acceleration-status", className="text-info")
                    ], className="control-group mt-3")
                ])
            ], className="controls-card")
        ], width=4)
    ], className="mb-4")


def create_memory_visualization():
    """Create the memory visualization panel."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Memory System"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div([
                                dbc.InputGroup([
                                    dbc.Input(id="memory-search", placeholder="Search memories..."),
                                    dbc.Button("Search", id="memory-search-button", color="secondary")
                                ], className="mb-3 mt-3"),
                                html.Div(id="memory-results", className="memory-results")
                            ])
                        ], label="Memory Search"),
                        dbc.Tab([
                            dcc.Graph(id="memory-stats-chart", config={'displayModeBar': False})
                        ], label="Memory Stats")
                    ])
                ])
            ], className="memory-card")
        ], width=12)
    ], className="mb-4")


def create_component_integration_visualization():
    """Create a visualization of component integration."""
    card = dbc.Card([
        dbc.CardHeader([
            html.H5("Component Integration", className="card-title"),
            html.Div([
                dbc.Button("Refresh", id="refresh-integration", size="sm", color="primary", className="me-1"),
            ], className="d-flex justify-content-end"),
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id="integration-network-graph",
                            figure=go.Figure(),
                            style={"height": "300px"},
                            config={"displayModeBar": False}
                        ),
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.H6("Integration Level"),
                            html.Div(id="integration-level", className="metric-value"),
                            
                            html.H6("Most Active Influences", className="mt-3"),
                            html.Div(id="active-influences", className="metric-list"),
                            
                            html.H6("Integration Events", className="mt-3"),
                            html.Div(id="integration-events", className="metric-value"),
                        ]),
                    ], width=4),
                ]),
            ]),
        ]),
    ])
    
    return card


# Assemble the layout
app.layout = dbc.Container([
    # State store for sharing data between callbacks
    dcc.Store(id="state-store", data={}),
    
    # Update interval
    dcc.Interval(id="update-interval", interval=5000, n_intervals=0),
    
    # Header
    create_header(),
    
    # Main dashboard content
    dbc.Row([
        # Left column: Status and charts
        dbc.Col([
            # Status cards
            create_status_cards(),
            
            # Development chart
            create_development_chart(),
            
            # Component Integration visualization
            create_component_integration_visualization(),
            
        ], width=7),
        
        # Right column: Interaction and memory panels
        dbc.Col([
            # Interaction panel
            create_interaction_panel(),
            
            # Memory visualization
            create_memory_visualization(),
        ], width=5),
    ]),
    
    # Acceleration Modal
    dbc.Modal([
        dbc.ModalHeader("Accelerating Development"),
        dbc.ModalBody(id="acceleration-modal-body", children=[
            html.P("Simulating development..."),
            dbc.Progress(id="acceleration-progress", value=0, animated=True, striped=True),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-acceleration-modal", className="ms-auto", n_clicks=0)
        ),
    ], id="acceleration-modal", is_open=False),
    
    # Footer
    html.Footer([
        html.Hr(),
        html.P("NeuralChild Dashboard - Psychological Mind Simulation", className="text-center"),
    ], className="mt-4"),
    
], fluid=True, className="p-4")


# Callbacks

@app.callback(
    Output("state-store", "data"),
    Input("update-interval", "n_intervals")
)
def update_state(n_intervals):
    """Update the state store with current system state."""
    if 'development_system' not in globals():
        return {}
    
    global development_system
    
    # Get essential state information
    child = development_system.child
    state = child.state
    
    # Create simplified state for dashboard
    dashboard_state = {
        "developmental_stage": state.developmental_stage.value,
        "developmental_substage": state.developmental_substage.value,
        "age_months": int(state.simulated_age_months),
        "vocabulary_size": len(state.vocabulary),
        "grammatical_complexity": state.metrics.grammatical_complexity,
        "emotional_regulation": state.metrics.emotional_regulation,
        "social_awareness": state.metrics.social_awareness,
        "abstract_thinking": state.metrics.abstract_thinking,
        "self_awareness": state.metrics.self_awareness,
        "emotions": [
            {"type": e.type.value, "intensity": e.intensity, "cause": e.cause}
            for e in state.current_emotional_state
        ],
        "metrics_history": {k: v[-20:] for k, v in state.metrics.history.items() if v},
        "component_states": {}
    }
    
    # Include transition state if available
    if state.stage_transition:
        dashboard_state["stage_transition"] = {
            "current_stage": state.stage_transition.current_stage.value,
            "next_stage": state.stage_transition.next_stage.value,
            "current_substage": state.stage_transition.current_substage.value,
            "next_substage": state.stage_transition.next_substage.value,
            "transition_progress": state.stage_transition.transition_progress,
            "metrics": state.stage_transition.metrics
        }
    
    # Include component states
    for component_id, component_state in state.component_states.items():
        dashboard_state["component_states"][component_id] = {
            "type": component_state.component_type,
            "activation": component_state.activation_level,
            "confidence": component_state.confidence
        }
    
    # Include integration metrics if available
    if hasattr(child, 'integration'):
        integration_metrics = child.integration.get_integration_metrics()
        dashboard_state["integration"] = integration_metrics
    
    return dashboard_state


@app.callback(
    [
        Output("stage-display", "children"),
        Output("age-display", "children"),
        Output("vocabulary-size", "children"),
        Output("grammatical-complexity", "children"),
        Output("emotional-regulation", "children"),
        Output("emotion-indicators", "children")
    ],
    Input("state-store", "data")
)
def update_status_cards(state):
    """Update the status cards with current state information."""
    if not state:
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    # Get developmental stage with proper formatting
    stage = state.get("developmental_stage", "Unknown")
    substage = state.get("developmental_substage", "Unknown")
    
    # Check if in transition
    in_transition = state.get("stage_transition") is not None
    
    # Format stage display
    stage_display = stage.replace("_", " ").title()
    substage_display = substage.replace("_", " ").title() if substage != "Unknown" else ""
    
    # Add transition indicator if applicable
    if in_transition:
        transition = state.get("stage_transition", {})
        next_stage = transition.get("next_stage", "").replace("_", " ").title()
        next_substage = transition.get("next_substage", "").replace("_", " ").title()
        progress = transition.get("transition_progress", 0) * 100
        
        stage_html = html.Div([
            html.Span(f"{stage_display}", className="stage-name"),
            html.Div([
                html.Span(f"→ {next_stage}", className="next-stage"),
                dbc.Progress(value=progress, className="transition-progress")
            ], className="transition-indicator")
        ])
        
        # Add substage info
        substage_html = html.Div([
            html.Span(f"{substage_display}", className="substage-name"),
            html.Span(f" → {next_substage}", className="next-substage")
        ], className="substage-display")
        
        # Combine stage and substage
        full_stage_display = html.Div([
            stage_html,
            substage_html
        ])
    else:
        # Regular stage display without transition
        full_stage_display = html.Div([
            html.Div(stage_display, className="stage-name"),
            html.Div(substage_display, className="substage-display")
        ])
    
    # Format age with proper units
    age_months = state.get("age_months", 0)
    if age_months < 24:
        age_display = f"{age_months} months"
    else:
        years = age_months // 12
        months = age_months % 12
        if months == 0:
            age_display = f"{years} year{'s' if years != 1 else ''}"
        else:
            age_display = f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}"
    
    # Format vocabulary size
    vocab_size = state.get("vocabulary_size", 0)
    
    # Format other metrics
    grammatical = state.get("grammatical_complexity", 0)
    grammatical_display = f"{grammatical:.2f}" if grammatical > 0 else "0.00"
    
    emotional = state.get("emotional_regulation", 0)
    emotional_display = f"{emotional:.2f}" if emotional > 0 else "0.00"
    
    # Create emotion indicators
    emotion_indicators = []
    for emotion in state.get("emotions", []):
        # Determine color based on emotion type
        color = "#6c757d"  # Default gray
        emotion_type = emotion.get("type", "")
        
        if emotion_type == "joy":
            color = "#ffc107"  # Yellow
        elif emotion_type == "sadness":
            color = "#0d6efd"  # Blue
        elif emotion_type == "anger":
            color = "#dc3545"  # Red
        elif emotion_type == "fear":
            color = "#6f42c1"  # Purple
        elif emotion_type == "surprise":
            color = "#20c997"  # Teal
        elif emotion_type == "disgust":
            color = "#198754"  # Green
        elif emotion_type == "trust":
            color = "#0dcaf0"  # Cyan
        elif emotion_type == "anticipation":
            color = "#fd7e14"  # Orange
        
        # Calculate size based on intensity
        intensity = emotion.get("intensity", 0.5)
        size = f"{30 + (intensity * 20)}px"
        
        # Create emotion indicator
        indicator = html.Div(
            className="emotion-indicator",
            style={
                "backgroundColor": color,
                "width": size,
                "height": size,
                "opacity": max(0.3, intensity)
            },
            title=f"{emotion_type.title()}: {intensity:.2f}"
        )
        
        emotion_indicators.append(indicator)
    
    return full_stage_display, age_display, vocab_size, grammatical_display, emotional_display, emotion_indicators


@app.callback(
    Output("development-chart", "figure"),
    Input("state-store", "data")
)
def update_development_chart(state):
    """Update the developmental metrics chart."""
    if not state or "metrics_history" not in state:
        # Create empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="No developmental data available",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            height=300
        )
        return fig
    
    # Get history data
    history = state["metrics_history"]
    
    # Create the main developmental chart
    fig = go.Figure()
    
    # Plot metrics
    colors = {
        "grammatical_complexity": "#0d6efd",  # Blue
        "emotional_regulation": "#dc3545",    # Red
        "social_awareness": "#fd7e14",        # Orange
        "object_permanence": "#0dcaf0",       # Cyan
        "abstract_thinking": "#6f42c1",       # Purple
        "self_awareness": "#20c997"           # Teal
    }
    
    # Get substage information for highlighting
    current_substage = state.get("developmental_substage", "")
    
    # Define developmental stages for x-axis highlights
    substage_ranges = {
        # Infancy
        "early_infancy": {"start": 0, "end": 8, "color": "rgba(255, 235, 235, 0.2)"},
        "middle_infancy": {"start": 8, "end": 16, "color": "rgba(255, 245, 235, 0.2)"},
        "late_infancy": {"start": 16, "end": 24, "color": "rgba(255, 255, 235, 0.2)"},
        
        # Early childhood
        "early_toddler": {"start": 24, "end": 36, "color": "rgba(245, 255, 235, 0.2)"},
        "late_toddler": {"start": 36, "end": 48, "color": "rgba(235, 255, 235, 0.2)"},
        "preschool": {"start": 48, "end": 60, "color": "rgba(235, 255, 245, 0.2)"},
        
        # Middle childhood
        "early_elementary": {"start": 60, "end": 84, "color": "rgba(235, 255, 255, 0.2)"},
        "middle_elementary": {"start": 84, "end": 108, "color": "rgba(235, 245, 255, 0.2)"},
        "late_elementary": {"start": 108, "end": 120, "color": "rgba(235, 235, 255, 0.2)"},
        
        # Adolescence
        "early_adolescence": {"start": 120, "end": 156, "color": "rgba(245, 235, 255, 0.2)"},
        "middle_adolescence": {"start": 156, "end": 192, "color": "rgba(255, 235, 255, 0.2)"},
        "late_adolescence": {"start": 192, "end": 216, "color": "rgba(255, 235, 245, 0.2)"},
        
        # Early adulthood
        "emerging_adult": {"start": 216, "end": 252, "color": "rgba(255, 235, 235, 0.2)"},
        "young_adult": {"start": 252, "end": 300, "color": "rgba(255, 245, 235, 0.2)"},
        "established_adult": {"start": 300, "end": 400, "color": "rgba(255, 255, 235, 0.2)"}
    }
    
    # Add rectangles for substage ranges
    for substage, range_info in substage_ranges.items():
        # Highlight current substage more prominently
        color = range_info["color"]
        line_width = 0
        
        if substage == current_substage:
            color = color.replace("0.2", "0.5")  # More visible
            line_width = 1
        
        fig.add_shape(
            type="rect",
            x0=range_info["start"],
            y0=0,
            x1=range_info["end"],
            y1=1,
            xref="x",
            yref="paper",
            fillcolor=color,
            opacity=1,
            layer="below",
            line_width=line_width,
            line=dict(color="rgba(0,0,0,0.3)")
        )
    
    # Add developmental stage dividers and labels
    stage_dividers = [
        {"month": 24, "label": "Early Childhood"},
        {"month": 60, "label": "Middle Childhood"},
        {"month": 120, "label": "Adolescence"},
        {"month": 216, "label": "Early Adulthood"}
    ]
    
    # Add vertical lines for major stage transitions
    for divider in stage_dividers:
        fig.add_shape(
            type="line",
            x0=divider["month"],
            y0=0,
            x1=divider["month"],
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color="rgba(0,0,0,0.5)",
                width=1,
                dash="dash"
            )
        )
        
        # Add stage label
        fig.add_annotation(
            x=divider["month"],
            y=1,
            text=divider["label"],
            showarrow=False,
            yshift=10,
            xshift=0,
            font=dict(
                size=10,
                color="rgba(0,0,0,0.7)"
            ),
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            borderpad=2
        )
    
    # Get current age
    current_age = state.get("age_months", 0)
    
    # Plot each metric
    for metric, values in history.items():
        if metric == "vocabulary_size":
            # Skip vocabulary size for this chart (different scale)
            continue
            
        # Create x-axis values
        max_points = len(values)
        # Generate age values (recent history, so work backwards from current age)
        x_values = [current_age - (max_points - i - 1) for i in range(max_points)]
        
        # Plot the line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=values,
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(color=colors.get(metric, "#6c757d")),
            hovertemplate='Age: %{x:.0f} months<br>%{y:.2f}<extra></extra>'
        ))
    
    # Add current age indicator
    fig.add_shape(
        type="line",
        x0=current_age,
        y0=0,
        x1=current_age,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="rgba(220,53,69,0.8)",
            width=2
        )
    )
    
    # Add current age label
    fig.add_annotation(
        x=current_age,
        y=0,
        text="Current Age",
        showarrow=False,
        yshift=-15,
        font=dict(
            size=10,
            color="rgba(220,53,69,1)"
        )
    )
    
    # Improve layout
    fig.update_layout(
        title="Developmental Progression",
        xaxis_title="Age (months)",
        yaxis_title="Metric Value (0-1)",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(
            range=[0, 1]
        ),
        xaxis=dict(
            range=[max(0, current_age - 30), current_age + 5]  # Show last 30 months + 5 months ahead
        ),
        hovermode="x unified"
    )
    
    # Add in-transition indicator if applicable
    if state.get("stage_transition"):
        transition = state.get("stage_transition", {})
        current_stage = transition.get("current_stage", "").replace("_", " ").title()
        next_stage = transition.get("next_stage", "").replace("_", " ").title()
        progress = transition.get("transition_progress", 0) * 100
        
        fig.add_annotation(
            x=current_age,
            y=0.95,
            text=f"Transitioning: {current_stage} → {next_stage} ({progress:.0f}%)",
            showarrow=False,
            font=dict(
                size=12,
                color="rgba(13,110,253,1)"
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(13,110,253,0.3)",
            borderwidth=1,
            borderpad=4,
            xanchor="right"
        )
    
    return fig


@app.callback(
    [
        Output("conversation-container", "children"),
        Output("user-input", "value")
    ],
    [
        Input("send-button", "n_clicks"),
        Input("user-input", "n_submit")
    ],
    [
        State("user-input", "value"),
        State("interaction-mode", "value"),
        State("mother-personality", "value"),
        State("conversation-container", "children")
    ]
)
def process_user_input(send_clicks, enter_pressed, input_text, mode, personality, current_conversation):
    """Process user input and update conversation."""
    global development_system, conversation_history
    
    # Initialize if needed
    if development_system is None:
        initialize_system()
    
    # Check for input trigger
    ctx = dash.callback_context
    if not ctx.triggered or (input_text is None or input_text.strip() == ""):
        return current_conversation or [], ""
    
    # Update mother personality if needed
    if personality != development_system.mother.personality.value:
        development_system.mother.set_personality(MotherPersonality(personality))
    
    # Process the input
    if mode == "user_mother_child":
        # User input to mother to child
        user_message = create_message("User", input_text, "user-message")
        
        # Simulate child's initial response if appropriate
        if development_system.child.state.developmental_stage == DevelopmentalStage.INFANCY:
            initial_vocalization = development_system.child.state.current_emotional_state[0].type.value[:3].lower()
            child_response, mother_response = development_system.simulate_interaction(initial_vocalization=initial_vocalization)
        else:
            child_response, mother_response = development_system.simulate_interaction(initial_text=input_text)
        
        # Create message components
        mother_text = mother_response.text
        mother_message = create_message("Mother", mother_text, "mother-message")
        
        if child_response.text:
            child_text = child_response.text
        else:
            child_text = f"*{child_response.vocalization}*"
        
        child_message = create_message("Child", child_text, "child-message")
        
        # Update conversation
        conversation_history.extend([user_message, mother_message, child_message])
        
    elif mode == "user_child":
        # Direct user input to child
        user_message = create_message("User", input_text, "user-message")
        
        # Create fake mother response (internal only)
        mother_response = MotherResponse(
            text=input_text,
            emotional_state=development_system.child.state.current_emotional_state
        )
        
        # Let child process this
        child_response = development_system.child.process_mother_response(mother_response)
        
        if child_response.text:
            child_text = child_response.text
        else:
            child_text = f"*{child_response.vocalization}*"
        
        child_message = create_message("Child", child_text, "child-message")
        
        # Update conversation
        conversation_history.extend([user_message, child_message])
    
    # Update developmental metrics
    development_system.child.update_developmental_metrics()
    development_system.child.check_stage_progression()
    
    # Keep only last 20 messages
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    
    return conversation_history, ""


@app.callback(
    Output("acceleration-modal", "is_open"),
    Output("acceleration-modal-body", "children"),
    Input("accelerate-button", "n_clicks"),
    Input("close-acceleration-modal", "n_clicks"),
    State("acceleration-months", "value"),
    State("acceleration-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_acceleration_modal(accelerate_clicks, close_clicks, months, is_open):
    """Toggle the acceleration modal and process acceleration if needed."""
    global development_system
    
    # Initialize if needed
    if development_system is None:
        initialize_system()
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, no_update
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "accelerate-button" and not is_open:
        # Start acceleration
        if months is None or months < 1:
            months = 1
        
        # Record initial state
        initial_stage = development_system.child.state.developmental_stage
        initial_age = int(development_system.child.state.simulated_age_months)
        
        # Perform acceleration
        stages_progressed = development_system.accelerate_development(months)
        
        # Record final state
        final_stage = development_system.child.state.developmental_stage
        final_age = int(development_system.child.state.simulated_age_months)
        
        # Create result message
        result = html.Div([
            html.P(f"Development accelerated by {months} months."),
            html.P(f"Age: {initial_age} months → {final_age} months"),
            html.P(f"Stage: {initial_stage.value.replace('_', ' ').title()} → {final_stage.value.replace('_', ' ').title()}")
        ])
        
        return True, result
    
    elif triggered_id == "close-acceleration-modal":
        return False, no_update
    
    return is_open, no_update


@app.callback(
    Output("memory-stats-chart", "figure"),
    Input("update-interval", "n_intervals")
)
def update_memory_stats(n_intervals):
    """Update the memory statistics chart."""
    global development_system
    
    # Initialize if needed
    if development_system is None:
        initialize_system()
    
    # Check if memory component exists
    memory_component = None
    for component_id, component in development_system.child.components.items():
        if "memory" in component_id.lower():
            memory_component = component
            break
    
    if memory_component is None:
        # Return placeholder
        fig = px.bar(
            x=["No memory data available"],
            y=[0],
            title="Memory Statistics"
        )
        return fig
    
    # Get memory stats
    try:
        stats = memory_component.get_memory_stats()
        
        # Create dataframe for the chart
        memory_types = ["Episodic", "Semantic"]
        counts = [stats.get("episodic_count", 0), stats.get("semantic_count", 0)]
        strengths = [
            stats.get("average_episodic_strength", 0) * 100,
            stats.get("average_semantic_confidence", 0) * 100
        ]
        
        # Create figure
        fig = go.Figure(data=[
            go.Bar(name="Memory Count", x=memory_types, y=counts, marker_color="#0d6efd"),
            go.Bar(name="Avg. Strength (%)", x=memory_types, y=strengths, marker_color="#fd7e14")
        ])
        
        # Update layout
        fig.update_layout(
            title="Memory System Statistics",
            xaxis_title="Memory Type",
            barmode="group",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        # Return placeholder
        fig = px.bar(
            x=["No memory data available"],
            y=[0],
            title="Memory Statistics"
        )
        return fig


@app.callback(
    Output("memory-results", "children"),
    Input("memory-search-button", "n_clicks"),
    State("memory-search", "value"),
    prevent_initial_call=True
)
def search_memories(n_clicks, query):
    """Search the child's memories."""
    global development_system
    
    if development_system is None or not query:
        return html.Div("No results found.")
    
    # Check if memory component exists
    memory_component = None
    for component_id, component in development_system.child.components.items():
        if "memory" in component_id.lower():
            memory_component = component
            break
    
    if memory_component is None:
        return html.Div("Memory system not initialized.")
    
    try:
        # Search memories
        stage = development_system.child.state.developmental_stage
        episodic_results = memory_component.retrieve_episodic_memories(
            query=query, 
            developmental_stage=stage,
            limit=5
        )
        semantic_results = memory_component.retrieve_semantic_memories(
            query=query,
            developmental_stage=stage,
            limit=5
        )
        
        # Create result cards
        result_cards = []
        
        if episodic_results:
            result_cards.append(html.H6("Episodic Memories"))
            for memory in episodic_results:
                result_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.P(memory.event_description, className="mb-1"),
                            html.Small(
                                f"Created: {memory.created_at.strftime('%Y-%m-%d')} | "
                                f"Strength: {memory.strength:.2f}",
                                className="text-muted"
                            )
                        ])
                    ], className="mb-2")
                )
        
        if semantic_results:
            result_cards.append(html.H6("Semantic Memories"))
            for memory in semantic_results:
                result_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.P(f"{memory.concept}: {memory.definition}", className="mb-1"),
                            html.Small(
                                f"Confidence: {memory.confidence:.2f} | "
                                f"Related: {', '.join(memory.related_concepts[:3])}",
                                className="text-muted"
                            )
                        ])
                    ], className="mb-2")
                )
        
        if not result_cards:
            result_cards.append(html.Div("No matching memories found."))
        
        return result_cards
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return html.Div(f"Error searching memories: {str(e)}")


@app.callback(
    Output("save-button", "disabled"),
    Output("save-button", "children"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True
)
def save_system_state(n_clicks):
    """Save the current system state."""
    global development_system
    
    if development_system is None:
        return False, "Save State"
    
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage = development_system.child.state.developmental_stage.value
        age = int(development_system.child.state.simulated_age_months)
        filename = f"neuralchild_{stage}_{age}m_{timestamp}.json"
        filepath = os.path.join(STATES_PATH, filename)
        
        # Save state
        development_system.save_system_state(filepath)
        
        # Temporarily disable button and show success message
        return True, "Saved!"
        
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return False, "Error Saving"


@app.callback(
    Output("save-button", "children", allow_duplicate=True),
    Output("save-button", "disabled", allow_duplicate=True),
    Input("update-interval", "n_intervals"),
    State("save-button", "children"),
    prevent_initial_call=True
)
def reset_save_button(n_intervals, current_text):
    """Reset the save button after a delay."""
    if current_text == "Saved!" or current_text == "Error Saving":
        return "Save State", False
    return no_update, no_update


@app.callback(
    [
        Output("integration-network-graph", "figure"),
        Output("integration-level", "children"),
        Output("active-influences", "children"),
        Output("integration-events", "children")
    ],
    [Input("state-store", "data"),
     Input("refresh-integration", "n_clicks")]
)
def update_integration_visualization(state, n_clicks):
    """Update the component integration visualization."""
    if not state or "integration" not in state:
        return (
            go.Figure(), 
            "No data", 
            "No data", 
            "No data"
        )
    
    integration = state["integration"]
    component_states = state.get("component_states", {})
    
    # Create nodes for each component
    nodes = []
    node_colors = []
    component_types = set()
    
    for component_id, component_data in component_states.items():
        component_type = component_data["type"]
        component_types.add(component_type)
        nodes.append(component_id)
        
        # Color nodes by component type
        if component_type == "memory":
            node_colors.append("rgba(65, 105, 225, 0.8)")  # Royal blue
        elif component_type == "language":
            node_colors.append("rgba(50, 205, 50, 0.8)")   # Lime green
        elif component_type == "emotional":
            node_colors.append("rgba(255, 99, 71, 0.8)")   # Tomato red
        elif component_type == "consciousness":
            node_colors.append("rgba(148, 0, 211, 0.8)")   # Dark violet
        elif component_type == "social":
            node_colors.append("rgba(255, 165, 0, 0.8)")   # Orange
        elif component_type == "cognitive":
            node_colors.append("rgba(30, 144, 255, 0.8)")  # Dodger blue
        else:
            node_colors.append("rgba(128, 128, 128, 0.8)") # Gray
    
    # Create edges between components
    source_indices = []
    target_indices = []
    edge_weights = []
    edge_colors = []
    
    # Use the average influences from integration data
    for connection, influence in integration.get("average_influences", {}).items():
        if influence < 0.05:  # Skip very weak connections
            continue
            
        source_type, target_type = connection.split("->")
        
        # Find component IDs matching these types
        for source_id, source_data in component_states.items():
            if source_data["type"] == source_type:
                for target_id, target_data in component_states.items():
                    if target_data["type"] == target_type and source_id != target_id:
                        source_idx = nodes.index(source_id)
                        target_idx = nodes.index(target_id)
                        
                        source_indices.append(source_idx)
                        target_indices.append(target_idx)
                        edge_weights.append(influence * 10)  # Scale for visibility
                        
                        # Color edges by influence strength
                        if influence > 0.6:
                            edge_colors.append("rgba(255, 0, 0, 0.6)")  # Strong: red
                        elif influence > 0.3:
                            edge_colors.append("rgba(255, 165, 0, 0.6)")  # Medium: orange
                        else:
                            edge_colors.append("rgba(0, 128, 0, 0.6)")  # Weak: green
    
    # Create network visualization
    fig = go.Figure()
    
    # Add edges as lines
    for i in range(len(source_indices)):
        source_idx = source_indices[i]
        target_idx = target_indices[i]
        weight = edge_weights[i]
        color = edge_colors[i]
        
        # For a simple directed edge, draw an arrow
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 0],
            mode='lines',
            line=dict(width=weight, color=color),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes as scatter points
    for i, node in enumerate(nodes):
        component_type = component_states[node]["type"]
        activation = component_states[node]["activation"]
        
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(
                size=20 + (activation * 10),  # Size varies with activation
                color=node_colors[i],
                line=dict(width=2, color='white')
            ),
            text=component_type.capitalize(),
            textposition="top center",
            name=node,
            hoverinfo='text',
            hovertext=f"{node}<br>Type: {component_type}<br>Activation: {activation:.2f}"
        ))
    
    # Improve layout
    fig.update_layout(
        title=None,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(248, 249, 250, 1)',
        paper_bgcolor='rgba(248, 249, 250, 1)',
        dragmode='pan'
    )
    
    # Create a force-directed layout
    import networkx as nx
    
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    
    for i in range(len(source_indices)):
        G.add_edge(nodes[source_indices[i]], nodes[target_indices[i]], weight=edge_weights[i])
    
    # Apply force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Update node positions
    for i, node in enumerate(nodes):
        if node in pos:
            fig.data[i + len(source_indices)].x = [pos[node][0]]
            fig.data[i + len(source_indices)].y = [pos[node][1]]
    
    # Update edge positions
    for i in range(len(source_indices)):
        source = nodes[source_indices[i]]
        target = nodes[target_indices[i]]
        if source in pos and target in pos:
            fig.data[i].x = [pos[source][0], pos[target][0]]
            fig.data[i].y = [pos[source][1], pos[target][1]]
    
    # Create integration level display
    integration_level = f"{integration.get('integration_level', 0):.2f}"
    
    # Create active influences list
    active_influences_html = []
    for connection, count in sorted(integration.get("integration_counts", {}).items(), key=lambda x: x[1], reverse=True)[:5]:
        active_influences_html.append(html.Div(f"{connection}: {count}", className="influence-item"))
    
    if not active_influences_html:
        active_influences_html = [html.Div("No active influences yet", className="text-muted")]
    
    # Create integration events count
    events_count = str(integration.get("total_integration_events", 0))
    
    return fig, integration_level, active_influences_html, events_count


# Helper functions

def create_message(sender, text, css_class):
    """Create a message component for the conversation."""
    return html.Div([
        html.Div(sender, className="message-sender"),
        html.Div(text, className="message-text")
    ], className=f"message {css_class}")


# CSS for the dashboard
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .dashboard-container {
                padding: 20px;
            }
            .dashboard-title {
                color: #343a40;
                font-weight: 600;
            }
            .header-row {
                align-items: center;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 15px;
            }
            .status-card, .chart-card, .interaction-card, .controls-card, .memory-card {
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .stage-value {
                font-weight: 600;
                color: #0d6efd;
            }
            .metric-item {
                margin-bottom: 8px;
            }
            .metric-value {
                font-weight: 600;
                color: #343a40;
            }
            .emotion-container {
                max-height: 150px;
                overflow-y: auto;
            }
            .emotion-indicator {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }
            .emotion-label {
                width: 80px;
                font-size: 0.85rem;
            }
            .emotion-bar-container {
                flex-grow: 1;
                background-color: #e9ecef;
                border-radius: 4px;
                height: 8px;
                margin: 0 10px;
            }
            .emotion-bar {
                height: 100%;
                border-radius: 4px;
            }
            .emotion-intensity {
                width: 40px;
                font-size: 0.85rem;
                text-align: right;
            }
            .conversation-container {
                height: 300px;
                overflow-y: auto;
                margin-bottom: 15px;
                padding: 10px;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            .message {
                margin-bottom: 10px;
                padding: 8px 12px;
                border-radius: 8px;
                max-width: 85%;
            }
            .user-message {
                background-color: #0d6efd;
                color: white;
                margin-left: auto;
            }
            .mother-message {
                background-color: #fd7e14;
                color: white;
            }
            .child-message {
                background-color: #20c997;
                color: white;
            }
            .message-sender {
                font-weight: bold;
                margin-bottom: 4px;
                font-size: 0.85rem;
            }
            .message-text {
                word-break: break-word;
            }
            .control-group {
                margin-bottom: 15px;
            }
            .control-label {
                font-weight: 600;
                margin-bottom: 5px;
                display: block;
            }
            .mode-selector, .personality-selector {
                width: 100%;
            }
            .memory-results {
                max-height: 250px;
                overflow-y: auto;
                padding: 5px;
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
"""


# Main entry point
if __name__ == "__main__":
    # Initialize the development system
    initialize_system()
    
    # Run the app
    app.run_server(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=(os.getenv("DEBUG", "False").lower() in ("true", "1", "t"))
    ) 