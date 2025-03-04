# app.py
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
from dash.exceptions import PreventUpdate
import threading
import uuid
import logging

# Import Neural Child components
from neural_child import NeuralChild, create_test_child
from networks.network_types import NetworkType
from language.developmental_stages import LanguageDevelopmentStage
from utils.visualization_data import (
    prepare_dashboard_data, 
    generate_network_graph_data,
    generate_emotion_chart_data,
    generate_vocabulary_chart_data,
    DashboardData
)
from config import get_config, init_config

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_child_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralChildDashboard")

# Initialize Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Neural Child - Large Mind Model"
server = app.server

# Constants
UPDATE_INTERVAL = 2000  # ms
MAX_CHAT_HISTORY = 100
DATA_DIR = Path("./data")
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LOGS_DIR = DATA_DIR / "logs"
METRICS_FILE = LOGS_DIR / "metrics.json"

# Ensure directories exist
for directory in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize Neural Child
config = init_config()
neural_child = None 
training_thread = None
training_active = False
stop_training = False
dashboard_data = None
chat_history = []
interactions_queue = []
last_update_time = datetime.now()
chat_available = False

# ===== Helper Functions =====

def format_date(date_string):
    """Format date string for display"""
    try:
        dt = datetime.fromisoformat(date_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return date_string

def initialize_neural_child():
    """Initialize the Neural Child instance"""
    global neural_child
    try:
        neural_child = create_test_child()
        logger.info("Neural Child initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Neural Child: {str(e)}")
        return False

def update_dashboard_data():
    """Update dashboard data from neural child state"""
    global dashboard_data, chat_available
    
    if neural_child is None:
        return None
    
    try:
        # Get all metrics and states from neural child
        network_states = neural_child.get_network_states()
        emotional_state = neural_child.emotional_state.model_dump()
        vocabulary_stats = neural_child.vocabulary_manager.get_vocabulary_statistics().model_dump()
        development_metrics = neural_child.metrics.model_dump()
        
        # System status
        system_status = {
            "age_days": neural_child.metrics.age_days,
            "vocabulary_size": neural_child.metrics.vocabulary_size,
            "developmental_stage": neural_child.metrics.development_stage.value,
            "total_interactions": neural_child.metrics.total_interactions,
            "attention_span": neural_child.metrics.attention_span,
            "emotional_stability": neural_child.metrics.emotional_stability,
            "training_active": training_active
        }
        
        # Check if chat is available (child is ready to interact)
        chat_available = neural_child.is_ready_for_chat()
        
        # Create dashboard data
        dashboard_data = prepare_dashboard_data(
            system_status=system_status,
            network_states=network_states,
            emotional_state=emotional_state,
            vocabulary_stats=vocabulary_stats,
            recent_interactions=neural_child.recent_interactions,
            development_metrics=development_metrics
        )
        
        # Log metrics periodically
        log_metrics(dashboard_data)
        
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Error updating dashboard data: {str(e)}")
        return None

def log_metrics(data: DashboardData):
    """Log metrics to file for historical tracking"""
    if not data:
        return
    
    # Extract key metrics for logging
    metrics = {
        "timestamp": data.timestamp,
        "age_days": data.system_status.get("age_days", 0),
        "vocabulary_size": data.system_status.get("vocabulary_size", 0),
        "developmental_stage": data.system_status.get("developmental_stage", "unknown"),
        "emotional_state": {k: v for k, v in data.emotional_state.model_dump().items() 
                            if k not in ["dominant_emotion"]},
        "vocabulary": {
            "total_words": data.vocabulary.total_words,
            "active_vocabulary": data.vocabulary.active_vocabulary,
            "passive_vocabulary": data.vocabulary.passive_vocabulary
        },
        "network_activations": {
            network: data.networks[network].activation 
            for network in data.networks
        }
    }
    
    # Append to metrics file
    try:
        with open(METRICS_FILE, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")

def start_training():
    """Start the training process in a separate thread"""
    global training_thread, training_active, stop_training
    
    if training_active:
        return
    
    stop_training = False
    training_active = True
    training_thread = threading.Thread(target=training_loop)
    training_thread.daemon = True
    training_thread.start()
    logger.info("Training started")

def stop_training_process():
    """Stop the training process"""
    global training_active, stop_training
    
    stop_training = True
    training_active = False
    logger.info("Training stopped")

def training_loop():
    """Main training loop for the Neural Child"""
    global neural_child, training_active, stop_training, interactions_queue
    
    if neural_child is None and not initialize_neural_child():
        training_active = False
        return
    
    logger.info("Training loop started")
    
    try:
        while training_active and not stop_training:
            # Process any queued interactions first
            if interactions_queue:
                user_message = interactions_queue.pop(0)
                process_chat_interaction(user_message)
            
            # Normal training interaction
            child_message, mother_message = neural_child.interact_with_mother()
            
            # Update chat history
            chat_history.append({
                "sender": "child",
                "message": child_message,
                "timestamp": datetime.now().isoformat()
            })
            chat_history.append({
                "sender": "mother",
                "message": mother_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim chat history if needed
            if len(chat_history) > MAX_CHAT_HISTORY:
                chat_history.pop(0)
            
            # Update dashboard data
            update_dashboard_data()
            
            # Sleep to avoid consuming too many resources
            time.sleep(0.5)
    
    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
    
    finally:
        training_active = False
        logger.info("Training loop ended")

def process_chat_interaction(user_message: str):
    """Process a chat interaction from the user"""
    global neural_child, chat_history
    
    if neural_child is None:
        return
    
    try:
        # Add user message to chat history
        chat_history.append({
            "sender": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send input to neural child's mother
        # Create a child state from the neural child
        child_state = neural_child.get_child_state()
        
        # The message from the user becomes part of the context for the mother's response
        mother_response = neural_child.mother.respond_to_child(child_state)
        
        # Process the mother's response through the neural child
        neural_child.process_mother_response(mother_response)
        
        # Generate a response from the neural child
        response = neural_child.generate_response()
        
        # Add child's response to chat history
        chat_history.append({
            "sender": "child",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim chat history if needed
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history.pop(0)
        
        # Update dashboard data
        update_dashboard_data()
        
    except Exception as e:
        logger.error(f"Error processing chat interaction: {str(e)}")
        # Add error message to chat history
        chat_history.append({
            "sender": "system",
            "message": f"Error processing your message: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

def save_checkpoint():
    """Save a checkpoint of the neural child state"""
    if neural_child is None:
        return None
    
    try:
        checkpoint_path = CHECKPOINTS_DIR / f"neural_child_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        neural_child.save_state(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        return None

def load_checkpoint(checkpoint_path: Path):
    """Load a checkpoint for the neural child"""
    global neural_child
    
    if neural_child is None:
        initialize_neural_child()
    
    try:
        neural_child.load_state(checkpoint_path)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return False

def get_available_checkpoints():
    """Get list of available checkpoints"""
    try:
        checkpoints = list(CHECKPOINTS_DIR.glob("neural_child_*.json"))
        return sorted(checkpoints, key=os.path.getmtime, reverse=True)
    except Exception as e:
        logger.error(f"Error getting checkpoints: {str(e)}")
        return []

def create_network_graph():
    """Create network graph visualization"""
    if dashboard_data is None:
        return []
    
    try:
        network_states = {
            network_type: network.model_dump() 
            for network_type, network in dashboard_data.networks.items()
        }
        graph_data = generate_network_graph_data(network_states)
        
        # Format for cytoscape
        elements = []
        
        # Add nodes
        for node in graph_data["nodes"]:
            elements.append({
                "data": {
                    "id": node["id"],
                    "label": node["label"],
                    "activation": node["activation"],
                    "confidence": node.get("confidence", 0),
                    "size": node["size"]
                },
                "classes": f"network-node {'active-node' if node['activation'] > 0.3 else ''}"
            })
        
        # Add edges
        for edge in graph_data["edges"]:
            elements.append({
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "strength": edge.get("strength", 0.5),
                    "type": edge.get("type", "connection")
                },
                "classes": f"connection-edge {edge.get('type', 'connection')}-edge"
            })
        
        return elements
    
    except Exception as e:
        logger.error(f"Error creating network graph: {str(e)}")
        return []

def create_emotion_radar_chart():
    """Create emotion radar chart visualization"""
    if dashboard_data is None:
        return {}
    
    try:
        emotional_state = dashboard_data.emotional_state
        chart_data = generate_emotion_chart_data(emotional_state)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=chart_data["values"],
            theta=chart_data["emotions"],
            fill='toself',
            name='Emotional State',
            line_color='rgba(75, 192, 192, 0.8)',
            fillcolor='rgba(75, 192, 192, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Emotional State (Dominant: {chart_data['dominant'].capitalize()} - {chart_data['dominant_intensity']:.2f})",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating emotion radar chart: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="No emotional data available")
        return fig

def create_vocabulary_charts():
    """Create vocabulary charts"""
    if dashboard_data is None:
        return {}
    
    try:
        vocabulary = dashboard_data.vocabulary
        chart_data = generate_vocabulary_chart_data(vocabulary)
        
        # Create vocabulary distribution chart
        dist_fig = go.Figure()
        
        dist_fig.add_trace(go.Bar(
            x=['Active', 'Passive', 'Total'],
            y=[chart_data["distribution"]["active"], 
               chart_data["distribution"]["passive"],
               chart_data["distribution"]["total"]],
            marker_color=['rgba(75, 192, 192, 0.8)', 
                         'rgba(153, 102, 255, 0.8)', 
                         'rgba(255, 159, 64, 0.8)']
        ))
        
        dist_fig.update_layout(
            title="Vocabulary Distribution",
            xaxis_title="Type",
            yaxis_title="Word Count",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        # Create category breakdown chart
        categories = chart_data["categories"]
        if categories:
            cat_fig = go.Figure()
            
            cat_fig.add_trace(go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Pastel)
            ))
            
            cat_fig.update_layout(
                title="Vocabulary by Category",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=30, r=30, t=50, b=30),
            )
        else:
            cat_fig = go.Figure()
            cat_fig.update_layout(
                title="No category data available",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        
        # Create skills chart
        skills = chart_data["skills"]
        skills_fig = go.Figure()
        
        skills_fig.add_trace(go.Bar(
            x=list(skills.keys()),
            y=list(skills.values()),
            marker_color=['rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)']
        ))
        
        skills_fig.update_layout(
            title="Language Skills",
            xaxis_title="Skill",
            yaxis_title="Proficiency (0-1)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        return {
            "distribution": dist_fig,
            "categories": cat_fig,
            "skills": skills_fig
        }
    
    except Exception as e:
        logger.error(f"Error creating vocabulary charts: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="No vocabulary data available")
        return {"distribution": fig, "categories": fig, "skills": fig}

def create_development_chart():
    """Create development metrics chart"""
    if dashboard_data is None:
        return {}
    
    try:
        metrics = dashboard_data.development_metrics
        
        # Create development metrics chart
        fig = go.Figure()
        
        # Add key metrics
        metrics_to_plot = {
            "attention_span": "Attention Span",
            "emotional_stability": "Emotional Stability",
            "social_awareness": "Social Awareness",
            "abstraction_capability": "Abstraction Capability"
        }
        
        for key, label in metrics_to_plot.items():
            if key in metrics:
                fig.add_trace(go.Bar(
                    x=[label],
                    y=[metrics[key]],
                    name=label
                ))
        
        fig.update_layout(
            title="Development Metrics",
            xaxis_title="Metric",
            yaxis_title="Value (0-1)",
            yaxis=dict(range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating development chart: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="No development data available")
        return fig

def create_network_activation_chart():
    """Create network activation chart"""
    if dashboard_data is None:
        return {}
    
    try:
        networks = dashboard_data.networks
        
        # Create activation chart
        fig = go.Figure()
        
        # Add activations
        network_names = []
        activations = []
        
        for network_type, network in networks.items():
            network_names.append(network.name)
            activations.append(network.activation)
        
        fig.add_trace(go.Bar(
            x=network_names,
            y=activations,
            marker_color='rgba(75, 192, 192, 0.8)'
        ))
        
        fig.update_layout(
            title="Neural Network Activations",
            xaxis_title="Network",
            yaxis_title="Activation (0-1)",
            yaxis=dict(range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=50, b=30),
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating network activation chart: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="No network data available")
        return fig

def format_chat_message(message):
    """Format a chat message for display"""
    sender = message.get("sender", "unknown")
    msg = message.get("message", "")
    timestamp = message.get("timestamp", "")
    
    # Format timestamp
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%H:%M:%S")
        except:
            formatted_time = timestamp
    else:
        formatted_time = ""
    
    # Determine message style
    if sender == "user":
        align = "right"
        bg_color = "primary"
        text_color = "white"
    elif sender == "child":
        align = "left"
        bg_color = "info"
        text_color = "white"
    elif sender == "mother":
        align = "left" 
        bg_color = "success"
        text_color = "white"
    else:
        align = "center"
        bg_color = "warning"
        text_color = "dark"
    
    return {
        "text": msg,
        "time": formatted_time,
        "sender": sender,
        "align": align,
        "bg_color": bg_color,
        "text_color": text_color
    }

# ===== App Layout =====

# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fas fa-brain mr-2", style={"fontSize": "24px"})),
                        dbc.Col(dbc.NavbarBrand("Neural Child - Large Mind Model", className="ml-2")),
                    ],
                    align="center",
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Start Training", 
                                      id="start-training-btn", 
                                      color="success", 
                                      className="mr-2"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Stop Training", 
                                      id="stop-training-btn", 
                                      color="danger", 
                                      className="mr-2"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Save Checkpoint", 
                                      id="save-checkpoint-btn", 
                                      color="info", 
                                      className="mr-2"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Load Checkpoint", 
                                      id="load-checkpoint-btn", 
                                      color="warning", 
                                      className="mr-2"),
                            width="auto",
                        ),
                    ],
                    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-3",
)

# Status Cards
status_cards = dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Age (Days)"),
            dbc.CardBody(html.H3(id="age-days-value", children="0.0"))
        ], color="primary", inverse=True, className="mb-3"),
        width=3
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Vocabulary Size"),
            dbc.CardBody(html.H3(id="vocabulary-size-value", children="0"))
        ], color="success", inverse=True, className="mb-3"),
        width=3
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Developmental Stage"),
            dbc.CardBody(html.H3(id="developmental-stage-value", children="Unknown"))
        ], color="info", inverse=True, className="mb-3"),
        width=3
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardHeader("Training Status"),
            dbc.CardBody(html.H3(id="training-status-value", children="Inactive"))
        ], color="warning", inverse=True, className="mb-3"),
        width=3
    ),
])

# Tabs Layout
tabs = dbc.Tabs(
    [
        dbc.Tab(label="Dashboard", tab_id="dashboard-tab"),
        dbc.Tab(label="Networks", tab_id="networks-tab"),
        dbc.Tab(label="Language", tab_id="language-tab"),
        dbc.Tab(label="Memory", tab_id="memory-tab"),
        dbc.Tab(label="Chat", tab_id="chat-tab"),
        dbc.Tab(label="Settings", tab_id="settings-tab"),
    ],
    id="tabs",
    active_tab="dashboard-tab",
)

# Dashboard Tab Content
dashboard_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Emotional State"),
            dbc.CardBody(dcc.Graph(id="emotion-radar-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("Development Metrics"),
            dbc.CardBody(dcc.Graph(id="development-metrics-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Neural Network Activations"),
            dbc.CardBody(dcc.Graph(id="network-activations-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("Vocabulary Distribution"),
            dbc.CardBody(dcc.Graph(id="vocabulary-distribution-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Recent Interactions"),
            dbc.CardBody(
                dash_table.DataTable(
                    id="recent-interactions-table",
                    columns=[
                        {"name": "Time", "id": "timestamp"},
                        {"name": "Child", "id": "child_message"},
                        {"name": "Mother", "id": "mother_message"}
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': '#303030',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#404040',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#383838'
                        }
                    ],
                    page_size=5
                )
            )
        ], className="mb-3"),
    ], width=12),
])

# Networks Tab Content
networks_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Neural Network Architecture"),
            dbc.CardBody([
                cyto.Cytoscape(
                    id='network-graph',
                    layout={'name': 'circle'},
                    style={'width': '100%', 'height': '600px'},
                    elements=[],
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'content': 'data(label)',
                                'background-color': '#1976d2',
                                'font-size': '12px',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'color': 'white',
                                'text-outline-width': 1,
                                'text-outline-color': '#1976d2',
                                'width': 'data(size)',
                                'height': 'data(size)'
                            }
                        },
                        {
                            'selector': '.active-node',
                            'style': {
                                'background-color': '#4caf50',
                                'text-outline-color': '#4caf50'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'width': 2,
                                'line-color': '#7f7f7f',
                                'target-arrow-color': '#7f7f7f',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier'
                            }
                        },
                        {
                            'selector': '.excitatory-edge',
                            'style': {
                                'line-color': '#4caf50',
                                'target-arrow-color': '#4caf50'
                            }
                        },
                        {
                            'selector': '.inhibitory-edge',
                            'style': {
                                'line-color': '#f44336',
                                'target-arrow-color': '#f44336'
                            }
                        },
                        {
                            'selector': '.modulatory-edge',
                            'style': {
                                'line-color': '#ff9800',
                                'target-arrow-color': '#ff9800',
                                'line-style': 'dashed'
                            }
                        },
                        {
                            'selector': '.feedback-edge',
                            'style': {
                                'line-color': '#9c27b0',
                                'target-arrow-color': '#9c27b0',
                                'line-style': 'dotted'
                            }
                        },
                        {
                            'selector': '.associative-edge',
                            'style': {
                                'line-color': '#2196f3',
                                'target-arrow-color': '#2196f3'
                            }
                        }
                    ]
                ),
                html.Div(id="network-info", className="mt-3")
            ])
        ], className="mb-3"),
    ], width=12),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Network Details"),
            dbc.CardBody(id="selected-network-details")
        ], className="mb-3"),
    ], width=12),
])

# Language Tab Content
language_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Vocabulary Statistics"),
            dbc.CardBody(dcc.Graph(id="vocabulary-skills-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("Recent Words Learned"),
            dbc.CardBody(
                html.Div(id="recent-words-learned")
            )
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Vocabulary Categories"),
            dbc.CardBody(dcc.Graph(id="vocabulary-categories-chart", config={'displayModeBar': False}))
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("Language Development Stage"),
            dbc.CardBody([
                dbc.Progress(id="language-stage-progress", value=0, className="mb-3"),
                html.Div(id="language-stage-description")
            ])
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Grammar Rules Development"),
            dbc.CardBody(
                dash_table.DataTable(
                    id="grammar-rules-table",
                    columns=[
                        {"name": "Rule", "id": "rule"},
                        {"name": "Stage", "id": "stage"},
                        {"name": "Mastery", "id": "mastery"}
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': '#303030',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#404040',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#383838'
                        }
                    ],
                    page_size=5
                )
            )
        ], className="mb-3"),
    ], width=12),
])

# Memory Tab Content
memory_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Memory Statistics"),
            dbc.CardBody([
                html.Div(id="memory-stats-content")
            ])
        ], className="mb-3"),
    ], width=12),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Long-Term Memory"),
            dbc.CardBody([
                dash_table.DataTable(
                    id="long-term-memory-table",
                    columns=[
                        {"name": "Concept", "id": "concept"},
                        {"name": "Type", "id": "type"},
                        {"name": "Importance", "id": "importance"},
                        {"name": "Age", "id": "age"}
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': '#303030',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#404040',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#383838'
                        }
                    ],
                    page_size=5
                )
            ])
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Working Memory"),
            dbc.CardBody([
                dash_table.DataTable(
                    id="working-memory-table",
                    columns=[
                        {"name": "Item", "id": "item"},
                        {"name": "Activation", "id": "activation"},
                        {"name": "State", "id": "state"}
                    ],
                    data=[],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': '#303030',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    style_header={
                        'backgroundColor': '#404040',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#383838'
                        }
                    ],
                    page_size=5
                )
            ])
        ], className="mb-3"),
    ], width=6),
])

# Chat Tab Content
chat_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader([
                html.Div("Chat with Neural Child", className="d-inline-block"),
                dbc.Badge(
                    "Not Ready",
                    id="chat-readiness-badge",
                    color="danger",
                    className="ml-2 float-right"
                )
            ]),
            dbc.CardBody([
                html.Div(
                    id="chat-history-container",
                    style={
                        "height": "500px",
                        "overflowY": "scroll",
                        "padding": "10px",
                        "border": "1px solid #444",
                        "borderRadius": "5px",
                        "marginBottom": "15px"
                    }
                ),
                dbc.InputGroup([
                    dbc.Input(
                        id="chat-input",
                        placeholder="Type your message...",
                        type="text",
                        disabled=True
                    ),
                    dbc.Button(
                        "Send",
                        id="send-message-btn",
                        color="primary",
                        disabled=True
                    )
                ]),
            ])
        ], className="mb-3"),
    ], width=8),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Current Child State"),
            dbc.CardBody([
                html.H5("Emotional State", className="card-title"),
                html.Div(id="chat-emotional-state", className="mb-3"),
                
                html.H5("Attention Focus", className="card-title"),
                html.Div(id="chat-attention-focus", className="mb-3"),
                
                html.H5("Active Drives", className="card-title"),
                html.Div(id="chat-active-drives", className="mb-3"),
                
                html.H5("Recent Thoughts", className="card-title"),
                html.Div(id="chat-recent-thoughts", className="mb-3"),
            ])
        ], className="mb-3"),
    ], width=4),
])

# Settings Tab Content
settings_tab = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Checkpoints"),
            dbc.CardBody([
                html.H5("Available Checkpoints"),
                html.Div(id="available-checkpoints-list", className="mb-3"),
                dbc.Button("Refresh Checkpoints", id="refresh-checkpoints-btn", color="primary", className="mr-2"),
            ])
        ], className="mb-3"),
    ], width=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Training Settings"),
            dbc.CardBody([
                html.H5("Simulation Speed"),
                dbc.InputGroup([
                    dbc.Input(
                        id="simulation-speed-input",
                        type="number",
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1.0
                    ),
                    dbc.Button(
                        "Apply",
                        id="apply-simulation-speed-btn",
                        color="primary"
                    )
                ], className="mb-3"),
                
                html.H5("Child Configuration"),
                html.Div(id="child-config-display"),
            ])
        ], className="mb-3"),
    ], width=6),
])

# Modals
checkpoint_modal = dbc.Modal(
    [
        dbc.ModalHeader("Select Checkpoint"),
        dbc.ModalBody([
            html.Div(id="checkpoint-list")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-checkpoint-modal-btn", className="ml-auto")
        ),
    ],
    id="checkpoint-modal",
    size="lg",
)

alert_container = html.Div(id="alert-container")

# Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        alert_container,
        status_cards,
        tabs,
        # Main tab content container
        html.Div(id="tab-content"),
        # Hidden tab contents with proper IDs
        html.Div(networks_tab, id="networks-tab-content", style={"display": "none"}),
        html.Div(language_tab, id="language-tab-content", style={"display": "none"}),
        html.Div(memory_tab, id="memory-tab-content", style={"display": "none"}),
        html.Div(chat_tab, id="chat-tab-content", style={"display": "none"}),
        html.Div(settings_tab, id="settings-tab-content", style={"display": "none"}),
        checkpoint_modal,
        dcc.Interval(
            id='interval-component',
            interval=UPDATE_INTERVAL,
            n_intervals=0
        ),
        dcc.Store(id='store-data', storage_type='session'),
    ], fluid=True),
])

# ===== Callbacks =====

@app.callback(
    [Output("tab-content", "children"),
     Output("networks-tab-content", "style"),
     Output("language-tab-content", "style"),
     Output("memory-tab-content", "style"),
     Output("chat-tab-content", "style"),
     Output("settings-tab-content", "style")],
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Show/hide content based on active tab"""
    hidden = {"display": "none"}
    visible = {"display": "block"}
    
    if active_tab == "dashboard-tab":
        return dashboard_tab, hidden, hidden, hidden, hidden, hidden
    elif active_tab == "networks-tab":
        return networks_tab, visible, hidden, hidden, hidden, hidden
    elif active_tab == "language-tab":
        return language_tab, hidden, visible, hidden, hidden, hidden
    elif active_tab == "memory-tab":
        return memory_tab, hidden, hidden, visible, hidden, hidden
    elif active_tab == "chat-tab":
        return chat_tab, hidden, hidden, hidden, visible, hidden
    elif active_tab == "settings-tab":
        return settings_tab, hidden, hidden, hidden, hidden, visible
    
    return html.P("This tab is not yet implemented"), hidden, hidden, hidden, hidden, hidden

@app.callback(
    [
        Output("age-days-value", "children"),
        Output("vocabulary-size-value", "children"),
        Output("developmental-stage-value", "children"),
        Output("training-status-value", "children"),
        Output("recent-interactions-table", "data"),
        Output("store-data", "data"),
    ],
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def update_status_cards(n_intervals):
    """Update status cards with latest data"""
    global dashboard_data, last_update_time
    
    # Update dashboard data if necessary
    current_time = datetime.now()
    if (current_time - last_update_time).total_seconds() > 1:
        dashboard_data = update_dashboard_data()
        last_update_time = current_time
    
    if dashboard_data is None:
        return "N/A", "N/A", "N/A", "Inactive", [], None
    
    # Prepare data for status cards
    age_days = f"{dashboard_data.system_status.get('age_days', 0):.1f}"
    vocabulary_size = str(dashboard_data.system_status.get("vocabulary_size", 0))
    developmental_stage = dashboard_data.system_status.get("developmental_stage", "unknown").replace("_", " ").title()
    training_status = "Active" if dashboard_data.system_status.get("training_active", False) else "Inactive"
    
    # Format recent interactions for table
    interactions_data = []
    for interaction in dashboard_data.recent_interactions:
        # Each interaction should have mother_verbal and child_message
        if 'mother_verbal' in interaction and 'child_message' in interaction:
            interactions_data.append({
                "timestamp": format_date(interaction.get("timestamp", "")),
                "child_message": interaction.get("child_message", ""),
                "mother_message": interaction.get("mother_verbal", "")
            })
    
    # Store dashboard data for other callbacks
    store_data = {
        "last_updated": current_time.isoformat(),
        "data_available": True
    }
    
    return age_days, vocabulary_size, developmental_stage, training_status, interactions_data, store_data

@app.callback(
    [
        Output("chat-readiness-badge", "children"),
        Output("chat-readiness-badge", "color"),
        Output("chat-input", "disabled"),
        Output("send-message-btn", "disabled")
    ],
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_chat_readiness(store_data):
    """Update chat readiness status"""
    if not store_data or not store_data.get("data_available", False):
        return "Not Ready", "danger", True, True
    
    if chat_available:
        return "Ready", "success", False, False
    else:
        return "Not Ready", "danger", True, True

@app.callback(
    [
        Output("emotion-radar-chart", "figure"),
        Output("network-activations-chart", "figure"),
        Output("vocabulary-distribution-chart", "figure"),
        Output("development-metrics-chart", "figure")
    ],
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_dashboard_charts(store_data):
    """Update dashboard charts"""
    if not store_data or not store_data.get("data_available", False):
        # Return empty charts
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Create emotion radar chart
    emotion_chart = create_emotion_radar_chart()
    
    # Create network activation chart
    network_chart = create_network_activation_chart()
    
    # Create vocabulary distribution chart
    vocab_charts = create_vocabulary_charts()
    vocab_dist_chart = vocab_charts.get("distribution", go.Figure())
    
    # Create development metrics chart
    development_chart = create_development_chart()
    
    return emotion_chart, network_chart, vocab_dist_chart, development_chart

@app.callback(
    [
        Output("vocabulary-skills-chart", "figure"),
        Output("vocabulary-categories-chart", "figure"),
        Output("recent-words-learned", "children"),
        Output("language-stage-progress", "value"),
        Output("language-stage-description", "children")
    ],
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_language_tab(store_data):
    """Update language tab content"""
    if not store_data or not store_data.get("data_available", False) or dashboard_data is None:
        # Return empty charts and content
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return empty_fig, empty_fig, "No data available", 0, "No data available"
    
    # Create vocabulary skills chart
    vocab_charts = create_vocabulary_charts()
    vocab_skills_chart = vocab_charts.get("skills", go.Figure())
    vocab_categories_chart = vocab_charts.get("categories", go.Figure())
    
    # Format recent words
    recent_words = dashboard_data.vocabulary.recent_words
    recent_words_content = []
    if recent_words:
        recent_words_content = [
            html.Div(
                dbc.Badge(word, color="info", className="mr-1 mb-1 p-2"),
                style={"display": "inline-block"}
            )
            for word in recent_words
        ]
    else:
        recent_words_content = [html.P("No recently learned words")]
    
    # Calculate language stage progress
    stage_values = {
        "pre_linguistic": 0,
        "holophrastic": 20,
        "telegraphic": 40,
        "simple_syntax": 60,
        "complex_syntax": 80,
        "advanced": 100
    }
    
    developmental_stage = dashboard_data.system_status.get("developmental_stage", "pre_linguistic")
    stage_progress = stage_values.get(developmental_stage, 0)
    
    # Stage descriptions
    stage_descriptions = {
        "pre_linguistic": "Babbling, crying, and non-linguistic vocalizations",
        "holophrastic": "Single words used to express complex ideas",
        "telegraphic": "2-3 word combinations without grammatical markers",
        "simple_syntax": "Simple sentences with basic grammar",
        "complex_syntax": "Complex sentences with subordinate clauses",
        "advanced": "Full language competence with abstract concepts"
    }
    
    stage_name = developmental_stage.replace("_", " ").title()
    stage_desc = stage_descriptions.get(developmental_stage, "")
    
    stage_description = [
        html.H5(f"Current Stage: {stage_name}"),
        html.P(stage_desc)
    ]
    
    return vocab_skills_chart, vocab_categories_chart, recent_words_content, stage_progress, stage_description

@app.callback(
    Output("network-graph", "elements"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_network_graph(store_data):
    """Update network graph visualization"""
    if not store_data or not store_data.get("data_available", False):
        return []
    
    elements = create_network_graph()
    return elements

@app.callback(
    Output("selected-network-details", "children"),
    Input("network-graph", "tapNodeData"),
    prevent_initial_call=True
)
def display_network_details(node_data):
    """Display details for selected network"""
    if not node_data or dashboard_data is None:
        return html.P("Select a network to view details")
    
    network_id = node_data.get("id")
    if network_id not in dashboard_data.networks:
        return html.P(f"No details available for network: {network_id}")
    
    network = dashboard_data.networks[network_id]
    
    # Format connections for display
    connections = []
    for target, conn_data in network.connections.items():
        connections.append(html.Li(f"{conn_data.get('target', target)} ({conn_data.get('type', 'connection')}, strength: {conn_data.get('strength', 0):.2f})"))
    
    return [
        html.H4(network.name),
        html.P(f"Type: {network.network_type}"),
        html.P(f"Activation: {network.activation:.2f}"),
        html.P(f"Confidence: {network.confidence:.2f}"),
        html.P(f"Training Progress: {network.training_progress:.2f}"),
        html.P(f"Error Rate: {network.error_rate:.2f}"),
        html.P(f"Last Active: {format_date(network.last_active)}"),
        html.H5("Connections:"),
        html.Ul(connections) if connections else html.P("No connections")
    ]

@app.callback(
    Output("grammar-rules-table", "data"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_grammar_rules_table(store_data):
    """Update grammar rules table"""
    if not store_data or not store_data.get("data_available", False) or neural_child is None:
        return []
    
    try:
        # Get grammar rules from syntactic processor
        grammar_rules = neural_child.syntactic_processor.grammar_rules
        
        # Format for table
        table_data = []
        for rule in grammar_rules:
            table_data.append({
                "rule": rule.name.replace("_", " ").title(),
                "stage": rule.min_stage.value.replace("_", " ").title(),
                "mastery": f"{rule.mastery_level:.2f}"
            })
        
        return table_data
    
    except Exception as e:
        logger.error(f"Error updating grammar rules table: {str(e)}")
        return []

@app.callback(
    Output("memory-stats-content", "children"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_memory_stats(store_data):
    """Update memory statistics"""
    if not store_data or not store_data.get("data_available", False) or neural_child is None:
        return html.P("No memory data available")
    
    try:
        # Get memory stats from different memory systems
        memory_stats = []
        
        # Add memory managers if they exist
        if hasattr(neural_child, 'vocabulary_manager') and neural_child.vocabulary_manager:
            vocab_stats = neural_child.vocabulary_manager.get_vocabulary_statistics()
            memory_stats.append(
                dbc.Col([
                    html.H5("Vocabulary Memory"),
                    html.P(f"Total Words: {vocab_stats.total_words}"),
                    html.P(f"Active Words: {vocab_stats.active_vocabulary}"),
                    html.P(f"Passive Words: {vocab_stats.passive_vocabulary}"),
                    html.P(f"Average Understanding: {vocab_stats.average_understanding:.2f}"),
                    html.P(f"Average Production: {vocab_stats.average_production:.2f}")
                ], width=4)
            )
        
        if not memory_stats:
            return html.P("No memory systems available")
        
        return dbc.Row(memory_stats)
    
    except Exception as e:
        logger.error(f"Error updating memory stats: {str(e)}")
        return html.P(f"Error loading memory statistics: {str(e)}")

@app.callback(
    Output("long-term-memory-table", "data"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_long_term_memory_table(store_data):
    """Update long-term memory table"""
    if not store_data or not store_data.get("data_available", False) or neural_child is None:
        return []
    
    try:
        # This is a simplified version since we don't have direct access to all memory systems
        # In a real implementation, you would get data from neural_child.memory_manager
        
        # Use vocabulary as a proxy for long-term memory
        if hasattr(neural_child, 'vocabulary_manager') and neural_child.vocabulary_manager:
            words = neural_child.vocabulary_manager.lexical_memory.words
            
            table_data = []
            for word, item in words.items():
                if item.understanding > 0.4:  # Only show words with reasonable understanding
                    age_days = (datetime.now() - item.learned_at).days
                    
                    table_data.append({
                        "concept": word,
                        "type": item.pos,
                        "importance": f"{item.recall_strength:.2f}",
                        "age": f"{age_days} days"
                    })
            
            # Sort by importance (recall strength)
            table_data.sort(key=lambda x: float(x["importance"]), reverse=True)
            
            return table_data[:10]  # Return top 10
        
        return []
    
    except Exception as e:
        logger.error(f"Error updating long-term memory table: {str(e)}")
        return []

@app.callback(
    Output("working-memory-table", "data"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_working_memory_table(store_data):
    """Update working memory table"""
    if not store_data or not store_data.get("data_available", False) or neural_child is None:
        return []
    
    try:
        # Simplified version since we don't have direct access to all memory systems
        # Use consciousness network as a proxy for working memory
        if NetworkType.CONSCIOUSNESS in neural_child.networks:
            consciousness = neural_child.networks[NetworkType.CONSCIOUSNESS]
            active_contents = consciousness._prepare_output_data()
            
            table_data = []
            
            # Active perceptions
            for item in active_contents.get("active_perceptions", []):
                table_data.append({
                    "item": item,
                    "activation": "High",
                    "state": "Perception"
                })
            
            # Active thoughts
            for item in active_contents.get("active_thoughts", []):
                table_data.append({
                    "item": item,
                    "activation": "Medium",
                    "state": "Thought"
                })
            
            # Self representations
            for item in active_contents.get("self_representations", []):
                table_data.append({
                    "item": item,
                    "activation": "High",
                    "state": "Self-Concept"
                })
            
            return table_data
        
        return []
    
    except Exception as e:
        logger.error(f"Error updating working memory table: {str(e)}")
        return []

@app.callback(
    Output("chat-history-container", "children"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_chat_history(store_data):
    """Update chat history display"""
    if not store_data or not store_data.get("data_available", False):
        return [html.P("No chat history available")]
    
    # Format chat messages
    chat_messages = []
    for message in chat_history:
        formatted = format_chat_message(message)
        
        message_style = {
            "padding": "10px",
            "borderRadius": "15px",
            "marginBottom": "10px",
            "maxWidth": "70%",
            "backgroundColor": f"var(--bs-{formatted['bg_color']})",
            "color": f"var(--bs-{formatted['text_color']})",
        }
        
        if formatted['align'] == "right":
            message_style["marginLeft"] = "auto"
        elif formatted['align'] == "center":
            message_style["marginLeft"] = "auto"
            message_style["marginRight"] = "auto"
            message_style["textAlign"] = "center"
        
        chat_messages.append(
            html.Div([
                html.Div(formatted['text'], style=message_style),
                html.Small(
                    f"{formatted['sender']} - {formatted['time']}",
                    style={
                        "display": "block",
                        "textAlign": formatted['align'],
                        "marginBottom": "5px",
                        "color": "#aaa"
                    }
                )
            ])
        )
    
    if not chat_messages:
        chat_messages = [html.P("No messages yet")]
    
    return chat_messages

@app.callback(
    [
        Output("chat-emotional-state", "children"),
        Output("chat-attention-focus", "children"),
        Output("chat-active-drives", "children"),
        Output("chat-recent-thoughts", "children")
    ],
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_chat_state_info(store_data):
    """Update chat state information panel"""
    if not store_data or not store_data.get("data_available", False) or dashboard_data is None:
        return "No data", "No data", "No data", "No data"
    
    # Emotional state
    emotional_state = dashboard_data.emotional_state
    dominant, intensity = emotional_state.dominant_emotion()
    emotional_content = [
        html.P(f"Dominant: {dominant.capitalize()} ({intensity:.2f})"),
        dbc.Progress(value=emotional_state.joy*100, className="mb-1", label=f"Joy: {emotional_state.joy:.2f}", color="success"),
        dbc.Progress(value=emotional_state.trust*100, className="mb-1", label=f"Trust: {emotional_state.trust:.2f}", color="info"),
        dbc.Progress(value=emotional_state.fear*100, className="mb-1", label=f"Fear: {emotional_state.fear:.2f}", color="warning"),
        dbc.Progress(value=emotional_state.sadness*100, className="mb-1", label=f"Sadness: {emotional_state.sadness:.2f}", color="danger"),
    ]
    
    # Attention focus
    attention_focus = []
    if neural_child and NetworkType.ATTENTION in neural_child.networks:
        attention = neural_child.networks[NetworkType.ATTENTION]
        attention_data = attention._prepare_output_data()
        focus_objects = attention_data.get("focus_objects", [])
        
        if focus_objects:
            attention_focus = [html.P("Currently focused on:")]
            for obj in focus_objects:
                attention_focus.append(dbc.Badge(obj, color="primary", className="mr-1 mb-1 p-2"))
        else:
            attention_focus = [html.P("Not focusing on anything specific")]
    else:
        attention_focus = [html.P("Attention data not available")]
    
    # Active drives
    active_drives = []
    if neural_child and NetworkType.DRIVES in neural_child.networks:
        drives = neural_child.networks[NetworkType.DRIVES]
        drives_data = drives._prepare_output_data()
        drive_levels = drives_data.get("drive_levels", {})
        
        if drive_levels:
            active_drives = []
            for drive, level in drive_levels.items():
                # Only show significant drives
                if level > 0.3:
                    active_drives.append(
                        dbc.Progress(
                            value=level*100, 
                            className="mb-1", 
                            label=f"{drive.capitalize()}: {level:.2f}", 
                            color="primary"
                        )
                    )
            
            if not active_drives:
                active_drives = [html.P("No significant drives active")]
        else:
            active_drives = [html.P("Drive data not available")]
    else:
        active_drives = [html.P("Drive data not available")]
    
    # Recent thoughts
    recent_thoughts = []
    if neural_child and NetworkType.THOUGHTS in neural_child.networks:
        thoughts = neural_child.networks[NetworkType.THOUGHTS]
        thought_data = thoughts._prepare_output_data()
        thoughts_list = thought_data.get("thoughts", [])
        
        if thoughts_list:
            recent_thoughts = []
            for thought in thoughts_list:
                recent_thoughts.append(html.Div(
                    thought, 
                    style={"padding": "5px", "marginBottom": "5px", "borderLeft": "3px solid var(--bs-primary)"}
                ))
        else:
            recent_thoughts = [html.P("No recent thoughts")]
    else:
        recent_thoughts = [html.P("Thought data not available")]
    
    return emotional_content, attention_focus, active_drives, recent_thoughts

@app.callback(
    Output("alert-container", "children"),
    [
        Input("start-training-btn", "n_clicks"),
        Input("stop-training-btn", "n_clicks"),
        Input("save-checkpoint-btn", "n_clicks")
    ],
    prevent_initial_call=True
)
def handle_training_controls(start_clicks, stop_clicks, save_clicks):
    """Handle training control buttons"""
    button_id = ctx.triggered_id if ctx.triggered_id else None
    
    if button_id == "start-training-btn":
        if neural_child is None and not initialize_neural_child():
            return dbc.Alert("Failed to initialize Neural Child", color="danger", dismissable=True, duration=4000)
        
        if training_active:
            return dbc.Alert("Training is already active", color="warning", dismissable=True, duration=2000)
        
        start_training()
        return dbc.Alert("Training started", color="success", dismissable=True, duration=2000)
    
    elif button_id == "stop-training-btn":
        if not training_active:
            return dbc.Alert("Training is not active", color="warning", dismissable=True, duration=2000)
        
        stop_training_process()
        return dbc.Alert("Training stopped", color="info", dismissable=True, duration=2000)
    
    elif button_id == "save-checkpoint-btn":
        if neural_child is None:
            return dbc.Alert("Neural Child not initialized", color="danger", dismissable=True, duration=4000)
        
        checkpoint_path = save_checkpoint()
        if checkpoint_path:
            return dbc.Alert(f"Checkpoint saved: {checkpoint_path.name}", color="success", dismissable=True, duration=4000)
        else:
            return dbc.Alert("Failed to save checkpoint", color="danger", dismissable=True, duration=4000)
    
    return no_update

@app.callback(
    Output("checkpoint-modal", "is_open"),
    [
        Input("load-checkpoint-btn", "n_clicks"),
        Input("close-checkpoint-modal-btn", "n_clicks")
    ],
    State("checkpoint-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_checkpoint_modal(load_clicks, close_clicks, is_open):
    """Toggle checkpoint modal dialog"""
    if load_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("checkpoint-list", "children"),
    Input("checkpoint-modal", "is_open"),
    prevent_initial_call=True
)
def update_checkpoint_list(is_open):
    """Update list of available checkpoints"""
    if not is_open:
        return []
    
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        return html.P("No checkpoints available")
    
    checkpoint_items = []
    for checkpoint in checkpoints:
        # Get checkpoint metadata if possible
        try:
            modified_time = datetime.fromtimestamp(os.path.getmtime(checkpoint))
            modified_time_str = modified_time.strftime("%Y-%m-%d %H:%M:%S")
            
            checkpoint_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.H5(checkpoint.name),
                        html.Small(f"Last modified: {modified_time_str}")
                    ], className="d-flex justify-content-between align-items-center"),
                    html.Div([
                        dbc.Button(
                            "Load",
                            id={"type": "load-checkpoint", "index": str(checkpoint)},
                            color="primary",
                            size="sm",
                            className="mt-2"
                        )
                    ])
                ])
            )
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint}: {str(e)}")
    
    return dbc.ListGroup(checkpoint_items)

@app.callback(
    Output("available-checkpoints-list", "children"),
    [Input("refresh-checkpoints-btn", "n_clicks")],
    prevent_initial_call=True
)
def refresh_available_checkpoints(n_clicks):
    """Refresh list of available checkpoints in settings tab"""
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        return html.P("No checkpoints available")
    
    checkpoint_items = []
    for checkpoint in checkpoints:
        # Get checkpoint metadata if possible
        try:
            modified_time = datetime.fromtimestamp(os.path.getmtime(checkpoint))
            modified_time_str = modified_time.strftime("%Y-%m-%d %H:%M:%S")
            
            checkpoint_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.H5(checkpoint.name),
                        html.Small(f"Last modified: {modified_time_str}")
                    ], className="d-flex justify-content-between align-items-center"),
                    html.Div([
                        dbc.Button(
                            "Load",
                            id={"type": "load-checkpoint-settings", "index": str(checkpoint)},
                            color="primary",
                            size="sm",
                            className="mt-2"
                        )
                    ])
                ])
            )
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint}: {str(e)}")
    
    return dbc.ListGroup(checkpoint_items)

@app.callback(
    Output("child-config-display", "children"),
    Input("store-data", "data"),
    prevent_initial_call=True
)
def update_child_config_display(store_data):
    """Update child configuration display"""
    if neural_child is None:
        return html.P("Neural Child not initialized")
    
    try:
        config = neural_child.config
        
        config_items = []
        
        # Learning parameters
        config_items.extend([
            html.H6("Learning Parameters"),
            html.P(f"Learning Rate Multiplier: {config.learning_rate_multiplier}"),
            html.P(f"Emotional Sensitivity: {config.emotional_sensitivity}"),
            html.P(f"Curiosity Factor: {config.curiosity_factor}"),
            html.P(f"Memory Retention: {config.memory_retention}"),
            html.Hr()
        ])
        
        # Development simulation
        config_items.extend([
            html.H6("Development Simulation"),
            html.P(f"Simulated Time Ratio: {config.simulated_time_ratio} days/day"),
            html.P(f"Chat Readiness Threshold: {config.chat_readiness_threshold} days"),
            html.Hr()
        ])
        
        # Network emphasis
        config_items.extend([
            html.H6("Network Emphasis"),
        ])
        
        for network_type, emphasis in config.network_emphasis.items():
            config_items.append(html.P(f"{network_type.value}: {emphasis:.2f}"))
        
        return config_items
    
    except Exception as e:
        logger.error(f"Error displaying child config: {str(e)}")
        return html.P(f"Error displaying configuration: {str(e)}")

@app.callback(
    Output("send-message-btn", "n_clicks"),
    [Input("chat-input", "n_submit"), Input("send-message-btn", "n_clicks")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def send_message(n_submit, n_clicks, message):
    """Send a message to the Neural Child"""
    if not message or not chat_available:
        raise PreventUpdate
    
    # Add to interactions queue
    interactions_queue.append(message)
    
    # Start training if not active to process message
    if not training_active:
        start_training()
    
    return 0  # Reset button clicks

@app.callback(
    Output("chat-input", "value"),
    [Input("send-message-btn", "n_clicks")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def clear_chat_input(n_clicks, value):
    """Clear chat input after sending"""
    if n_clicks:
        return ""
    return value

# Callback for loading a checkpoint from the modal
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.location.reload();
            return true;
        }
        return false;
    }
    """,
    Output("checkpoint-modal", "id"),
    Input({"type": "load-checkpoint", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)

# Callback for loading a checkpoint from the settings tab
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.location.reload();
            return true;
        }
        return false;
    }
    """,
    Output("refresh-checkpoints-btn", "n_clicks"),
    Input({"type": "load-checkpoint-settings", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)

# Callback to handle simulation speed changes
@app.callback(
    Output("alert-container", "children", allow_duplicate=True),
    [Input("apply-simulation-speed-btn", "n_clicks")],
    [State("simulation-speed-input", "value")],
    prevent_initial_call=True
)
def apply_simulation_speed(n_clicks, speed):
    """Apply new simulation speed"""
    if not n_clicks or speed is None:
        raise PreventUpdate
    
    if neural_child:
        try:
            neural_child.simulation_speed = float(speed)
            return dbc.Alert(f"Simulation speed set to {speed}", color="success", dismissable=True, duration=2000)
        except Exception as e:
            logger.error(f"Error setting simulation speed: {str(e)}")
            return dbc.Alert(f"Error setting simulation speed: {str(e)}", color="danger", dismissable=True, duration=4000)
    
    return dbc.Alert("Neural Child not initialized", color="warning", dismissable=True, duration=2000)

# ===== Run Server =====

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except KeyboardInterrupt:
        # Clean shutdown
        if training_active:
            stop_training_process()
        
        if training_thread and training_thread.is_alive():
            training_thread.join(timeout=2.0)
        
        logger.info("Dashboard shut down")