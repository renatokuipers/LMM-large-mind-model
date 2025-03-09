#!/usr/bin/env python3
"""
Dashboard application for the Large Mind Model (LMM) project.

This dashboard provides visualization and control for the LMM development process,
showing cognitive module development, conversations with the mother, neural activity,
and other key metrics in an interactive interface.
"""

import os
import time
import threading
import logging
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import torch

# Core project components
from lmm_project.core.mind import Mind
from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message

# Mother interface
from lmm_project.interfaces.mother.mother_llm import MotherLLM

# Visualization utilities
from lmm_project.utils.visualization import visualize_development

# Import main system functions to reuse
from lmm_project.main import (
    initialize_mind, 
    initialize_mother,
    initialize_neural_substrate,
    initialize_learning_engines,
    initialize_database,
    process_interaction,
    generate_lmm_thought
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "dashboard.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress some noisy loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Global variables for system state
TRAINING_RUNNING = False
TRAINING_PAUSED = False
TRAINING_THREAD = None
EVENT_BUS = EventBus()
STATE_MANAGER = StateManager()
DB_CONN = None
CURRENT_SESSION_ID = None
LMM_COMPONENTS = {
    "mind": None,
    "mother": None,
    "neural_substrate": None,
    "learning_engines": None
}

# Dark theme colors
COLORS = {
    "background": "#121212",
    "card_bg": "#1E1E1E",
    "accent": "#BB86FC",
    "accent_dark": "#7C4DFF",
    "secondary": "#03DAC6",
    "text": "#FFFFFF",
    "text_secondary": "#B0B0B0",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#CF6679",
    "info": "#2196F3",
    "neural_colors": px.colors.qualitative.Plotly,
    "module_colors": px.colors.qualitative.Pastel,
    "stage_colors": {
        "prenatal": "#9575CD",
        "infant": "#7986CB",
        "child": "#64B5F6",
        "adolescent": "#4FC3F7",
        "adult": "#4DD0E1"
    }
}

# Initialize Dash app with Bootstrap for styling
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # CYBORG is a dark theme
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "LMM Development Dashboard"

# Custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom CSS */
            body {
                background-color: #121212;
                color: #FFFFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card {
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                background-color: #1E1E1E;
                margin-bottom: 20px;
                border: none;
            }
            .card-header {
                background-color: rgba(187, 134, 252, 0.1);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: 600;
            }
            .btn {
                border-radius: 30px;
                padding: 8px 20px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.2s;
            }
            .btn-primary {
                background-color: #BB86FC;
                border-color: #BB86FC;
            }
            .btn-primary:hover {
                background-color: #7C4DFF;
                border-color: #7C4DFF;
            }
            .btn-success {
                background-color: #00C853;
                border-color: #00C853;
            }
            .btn-warning {
                background-color: #FFD600;
                border-color: #FFD600;
                color: #121212;
            }
            .btn-danger {
                background-color: #CF6679;
                border-color: #CF6679;
            }
            .btn-info {
                background-color: #03DAC6;
                border-color: #03DAC6;
                color: #121212;
            }
            /* Neural network visualization */
            .neural-node {
                fill: #BB86FC;
                stroke: #7C4DFF;
                stroke-width: 2px;
            }
            .neural-connection {
                stroke: rgba(3, 218, 198, 0.4);
                stroke-width: 1.5px;
            }
            /* Timeline styling */
            .timeline-item {
                border-left: 3px solid #BB86FC;
                padding-left: 15px;
                margin-bottom: 10px;
            }
            /* Conversation styling */
            .lmm-message {
                background-color: rgba(187, 134, 252, 0.1);
                border-radius: 15px 15px 15px 0;
                padding: 10px 15px;
                margin-bottom: 10px;
                max-width: 80%;
                align-self: flex-start;
            }
            .mother-message {
                background-color: rgba(3, 218, 198, 0.1);
                border-radius: 15px 15px 0 15px;
                padding: 10px 15px;
                margin-bottom: 10px;
                max-width: 80%;
                align-self: flex-end;
            }
            .conversation-container {
                display: flex;
                flex-direction: column;
                height: 300px;
                overflow-y: auto;
                padding: 15px;
                background-color: #121212;
                border-radius: 12px;
            }
            /* Progress bars */
            .progress {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 30px;
                margin-bottom: 10px;
            }
            .progress-bar {
                border-radius: 30px;
            }
            /* Module cards */
            .module-card {
                transition: transform 0.2s;
            }
            .module-card:hover {
                transform: translateY(-5px);
            }
            /* Status indicator */
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }
            .status-running {
                background-color: #00C853;
                box-shadow: 0 0 10px #00C853;
            }
            .status-paused {
                background-color: #FFD600;
                box-shadow: 0 0 10px #FFD600;
            }
            .status-stopped {
                background-color: #CF6679;
                box-shadow: 0 0 10px #CF6679;
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

# Define the layout of the dashboard
app.layout = html.Div([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Large Mind Model Dashboard", 
                    style={"margin-top": "20px", "margin-bottom": "5px"}),
            html.P("Cognitive Development Monitoring System",
                   style={"color": COLORS["text_secondary"], "margin-bottom": "20px"}),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span(id="status-indicator", className="status-indicator status-stopped"),
                html.Span(id="status-text", children="Stopped", style={"vertical-align": "middle"})
            ], style={"margin-top": "30px", "margin-bottom": "10px"}),
            dbc.ButtonGroup([
                dbc.Button("Start", id="start-btn", color="success", 
                           className="me-2", style={"width": "100px"}),
                dbc.Button("Pause", id="pause-btn", color="warning", 
                           className="me-2", style={"width": "100px"}, disabled=True),
                dbc.Button("Stop", id="stop-btn", color="danger", 
                           style={"width": "100px"}, disabled=True),
            ], style={"margin-bottom": "20px"}),
        ], width=4, className="text-end"),
    ]),
    
    # System information row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Info"),
                dbc.CardBody([
                    html.Div(id="system-info"),
                    dcc.Interval(
                        id='system-info-interval',
                        interval=2000,  # 2 seconds
                        n_intervals=0
                    ),
                ])
            ], className="card")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Status"),
                dbc.CardBody([
                    html.Div(id="development-status")
                ])
            ], className="card")
        ], width=8),
    ]),
    
    # Developmental progression chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Developmental Progression"),
                dbc.CardBody([
                    dcc.Graph(id="development-graph", style={"height": "300px"}),
                    dcc.Interval(
                        id='development-graph-interval',
                        interval=10000,  # 10 seconds
                        n_intervals=0
                    ),
                ])
            ], className="card")
        ], width=12),
    ]),
    
    # Conversation and neural activity
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Conversation History"),
                dbc.CardBody([
                    html.Div(id="conversation-container", className="conversation-container"),
                    dcc.Interval(
                        id='conversation-interval',
                        interval=5000,  # 5 seconds
                        n_intervals=0
                    ),
                ])
            ], className="card")
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Neural Activity"),
                dbc.CardBody([
                    dcc.Graph(id="neural-activity-graph", style={"height": "300px"}),
                    dcc.Interval(
                        id='neural-activity-interval',
                        interval=5000,  # 5 seconds
                        n_intervals=0
                    ),
                ])
            ], className="card")
        ], width=5),
    ]),
    
    # Module development cards
    dbc.Row([
        dbc.Col([
            html.H4("Cognitive Module Development", 
                    style={"margin-top": "20px", "margin-bottom": "15px"}),
        ], width=12),
    ]),
    
    dbc.Row(id="module-cards"),
    
    # Detailed metrics tabs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detailed Metrics"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="Language", tab_id="tab-language", children=[
                            html.Div(id="language-metrics", style={"padding": "15px"})
                        ]),
                        dbc.Tab(label="Memory", tab_id="tab-memory", children=[
                            html.Div(id="memory-metrics", style={"padding": "15px"})
                        ]),
                        dbc.Tab(label="Perception", tab_id="tab-perception", children=[
                            html.Div(id="perception-metrics", style={"padding": "15px"})
                        ]),
                        dbc.Tab(label="Attention", tab_id="tab-attention", children=[
                            html.Div(id="attention-metrics", style={"padding": "15px"})
                        ]),
                        dbc.Tab(label="Learning", tab_id="tab-learning", children=[
                            html.Div(id="learning-metrics", style={"padding": "15px"})
                        ]),
                    ], id="metrics-tabs", active_tab="tab-language"),
                    dcc.Interval(
                        id='metrics-interval',
                        interval=8000,  # 8 seconds
                        n_intervals=0
                    ),
                ])
            ], className="card", style={"margin-top": "20px"})
        ], width=12),
    ]),
    
    # Hidden divs for storing state
    html.Div(id='training-state-store', style={'display': 'none'}),
    dcc.Store(id='session-data-store'),
    
    # Main update interval
    dcc.Interval(
        id='main-interval',
        interval=1000,  # 1 second
        n_intervals=0
    ),
])

# Helper functions
def get_db_connection():
    """Get a connection to the database"""
    try:
        conn = sqlite3.connect(os.path.join("lmm_project", "storage", "lmm_data.db"))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def get_session_data(session_id=None):
    """Get data for the specified session or the latest session"""
    try:
        conn = get_db_connection()
        if conn is None:
            return None
            
        cursor = conn.cursor()
        
        if session_id:
            # Get specified session
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        else:
            # Get latest session
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 1")
            
        session = cursor.fetchone()
        
        if not session:
            return None
            
        # Convert to dict
        session_data = dict(session)
        
        # Get interactions for this session
        cursor.execute("""
            SELECT * FROM interactions 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_data['id'],))
        
        interactions = [dict(row) for row in cursor.fetchall()]
        session_data['interactions'] = interactions
        
        # Get mind states for this session
        cursor.execute("""
            SELECT * FROM mind_states 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_data['id'],))
        
        mind_states = [dict(row) for row in cursor.fetchall()]
        session_data['mind_states'] = mind_states
        
        conn.close()
        return session_data
        
    except Exception as e:
        logger.error(f"Error getting session data: {e}")
        if conn:
            conn.close()
        return None

def start_training_thread():
    """Start a new training thread"""
    global TRAINING_THREAD, TRAINING_RUNNING, TRAINING_PAUSED, CURRENT_SESSION_ID
    
    if TRAINING_THREAD and TRAINING_THREAD.is_alive():
        return False
    
    # Create necessary directories
    os.makedirs(os.path.join("lmm_project", "logs"), exist_ok=True)
    os.makedirs(os.path.join("lmm_project", "storage"), exist_ok=True)
    os.makedirs(os.path.join("lmm_project", "visualization", "output"), exist_ok=True)
    
    # Initialize the LMM components
    EVENT_BUS = EventBus()
    STATE_MANAGER = StateManager()
    
    LMM_COMPONENTS["mind"] = initialize_mind(EVENT_BUS)
    LMM_COMPONENTS["mother"] = initialize_mother(personality_profile="nurturing")
    LMM_COMPONENTS["neural_substrate"] = initialize_neural_substrate(EVENT_BUS)
    LMM_COMPONENTS["learning_engines"] = initialize_learning_engines(EVENT_BUS)
    
    # Generate session ID
    CURRENT_SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize database and session
    DB_CONN = initialize_database()
    update_session_info(DB_CONN, CURRENT_SESSION_ID, LMM_COMPONENTS["mind"])
    
    # Start the training thread
    TRAINING_THREAD = threading.Thread(target=training_loop, daemon=True)
    TRAINING_RUNNING = True
    TRAINING_PAUSED = False
    TRAINING_THREAD.start()
    
    return True

def stop_training_thread():
    """Stop the training thread"""
    global TRAINING_RUNNING, TRAINING_PAUSED
    
    TRAINING_RUNNING = False
    TRAINING_PAUSED = False
    
    # Final update to session
    if DB_CONN and CURRENT_SESSION_ID and LMM_COMPONENTS["mind"]:
        update_session_info(DB_CONN, CURRENT_SESSION_ID, LMM_COMPONENTS["mind"], is_finished=True)
        
    return True

def pause_training_thread():
    """Pause the training thread"""
    global TRAINING_PAUSED
    
    TRAINING_PAUSED = not TRAINING_PAUSED
    return TRAINING_PAUSED

def update_session_info(conn, session_id, mind, is_finished=False, interaction_count=0):
    """Update session information in the database"""
    try:
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        current_time = datetime.now().isoformat()
        
        if not session:
            # Create new session
            cursor.execute('''
            INSERT INTO sessions (
                id, start_time, end_time, initial_age, final_age, 
                interactions_count, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                current_time,
                current_time if is_finished else None,
                mind.age,
                mind.age,
                interaction_count,
                f"Development session from {mind.developmental_stage} stage"
            ))
        else:
            # Update existing session
            cursor.execute('''
            UPDATE sessions SET
                end_time = ?,
                final_age = ?,
                interactions_count = ?
            WHERE id = ?
            ''', (
                current_time if is_finished else None,
                mind.age,
                interaction_count,
                session_id
            ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating session info: {e}")

def save_interaction_to_db(conn, result, interaction_id, session_id):
    """Save interaction details to the database"""
    try:
        cursor = conn.cursor()
        
        # Get perception information if available
        perception_stats = {}
        if "perception" in result and "perception_result" in result["perception"]:
            pr = result["perception"]["perception_result"]
            perception_stats = {
                "pattern_count": len(pr.get("detected_patterns", [])),
                "novelty_score": pr.get("novelty_score"),
                "intensity_score": pr.get("intensity_score")
            }
        
        # Insert into database
        cursor.execute('''
        INSERT INTO interactions (
            session_id, timestamp, lmm_input, mother_response, audio_path,
            developmental_stage, age, development_increment, 
            perception_stats, interaction_details
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.now().isoformat(),
            result.get("lmm_input", ""),
            result.get("mother_response", {}).get("text", ""),
            result.get("mother_response", {}).get("audio_path", ""),
            result.get("mind_state", {}).get("developmental_stage", ""),
            result.get("mind_state", {}).get("age", 0),
            result.get("development_increment", 0),
            json.dumps(perception_stats),
            json.dumps(result.get("mother_response", {}).get("interaction_details", {}))
        ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving interaction to DB: {e}")

def save_state_to_db(conn, mind, session_id, emotional_state="neutral"):
    """Save mind state to the database"""
    try:
        cursor = conn.cursor()
        
        # Gather module states
        module_states = {
            module_type: {
                "development_level": module.development_level,
            } for module_type, module in mind.modules.items()
        }
        
        # Insert into database
        cursor.execute('''
        INSERT INTO mind_states (
            timestamp, session_id, age, developmental_stage, 
            module_states, emotional_state
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            session_id,
            mind.age,
            mind.developmental_stage,
            json.dumps(module_states),
            emotional_state
        ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving state to DB: {e}")

def training_loop():
    """Main training loop that runs in a separate thread"""
    global TRAINING_RUNNING, TRAINING_PAUSED, DB_CONN, CURRENT_SESSION_ID
    
    logger.info(f"Starting training loop for session {CURRENT_SESSION_ID}")
    
    mind = LMM_COMPONENTS["mind"]
    mother = LMM_COMPONENTS["mother"]
    neural_substrate = LMM_COMPONENTS["neural_substrate"]
    learning_engines = LMM_COMPONENTS["learning_engines"]
    
    interaction_count = 0
    context = {"previous_interactions": []}
    
    try:
        while TRAINING_RUNNING:
            # Check if paused
            if TRAINING_PAUSED:
                time.sleep(0.5)
                continue
                
            interaction_count += 1
            logger.info(f"Processing interaction {interaction_count}")
            
            # Generate a thought from the LMM
            lmm_thought = generate_lmm_thought(mind, context)
            
            # Process the interaction
            result = process_interaction(
                mind=mind,
                mother=mother,
                neural_substrate=neural_substrate,
                learning_engines=learning_engines,
                lmm_input=lmm_thought,
                state_manager=STATE_MANAGER
            )
            
            # Update context
            mother_response = result["mother_response"]["text"]
            context["previous_interactions"].append({
                "lmm": lmm_thought,
                "mother": mother_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save interaction to database
            save_interaction_to_db(DB_CONN, result, interaction_count, CURRENT_SESSION_ID)
            
            # Save state periodically (every 5 interactions)
            if interaction_count % 5 == 0:
                emotional_state = STATE_MANAGER.get_state("emotional_state") or "neutral"
                save_state_to_db(DB_CONN, mind, CURRENT_SESSION_ID, emotional_state)
                
                # Update session info
                update_session_info(DB_CONN, CURRENT_SESSION_ID, mind, False, interaction_count)
            
            # Sleep to prevent high CPU usage
            time.sleep(3)
            
    except Exception as e:
        logger.error(f"Error in training loop: {e}")
    finally:
        # Final update to session
        if CURRENT_SESSION_ID:
            update_session_info(DB_CONN, CURRENT_SESSION_ID, mind, is_finished=True, interaction_count=interaction_count)
        
        logger.info("Training loop ended")

# Callbacks
@app.callback(
    [Output('start-btn', 'disabled'),
     Output('pause-btn', 'disabled'),
     Output('stop-btn', 'disabled'),
     Output('status-indicator', 'className'),
     Output('status-text', 'children')],
    [Input('start-btn', 'n_clicks'),
     Input('pause-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('main-interval', 'n_intervals')],
    prevent_initial_call=True
)
def handle_control_buttons(start_clicks, pause_clicks, stop_clicks, n_intervals):
    """Handle button clicks for controlling the training process"""
    global TRAINING_RUNNING, TRAINING_PAUSED
    
    trigger = ctx.triggered_id
    
    if trigger == 'start-btn' and start_clicks:
        success = start_training_thread()
        if success:
            return True, False, False, "status-indicator status-running", "Running"
    
    elif trigger == 'pause-btn' and pause_clicks:
        paused = pause_training_thread()
        if paused:
            return True, False, False, "status-indicator status-paused", "Paused"
        else:
            return True, False, False, "status-indicator status-running", "Running"
    
    elif trigger == 'stop-btn' and stop_clicks:
        success = stop_training_thread()
        if success:
            return False, True, True, "status-indicator status-stopped", "Stopped"
    
    # For interval updates, just return current state
    if TRAINING_RUNNING:
        if TRAINING_PAUSED:
            return True, False, False, "status-indicator status-paused", "Paused"
        else:
            return True, False, False, "status-indicator status-running", "Running"
    else:
        return False, True, True, "status-indicator status-stopped", "Stopped"

@app.callback(
    Output('system-info', 'children'),
    Input('system-info-interval', 'n_intervals')
)
def update_system_info(n_intervals):
    """Update system information display"""
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    cuda_info = f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "N/A"
    
    # Get components info if initialized
    modules = "Not initialized"
    if LMM_COMPONENTS["mind"]:
        modules = ", ".join(list(LMM_COMPONENTS["mind"].modules.keys()))
    
    mother_info = "Not initialized"
    if LMM_COMPONENTS["mother"]:
        mother_info = f"{LMM_COMPONENTS['mother'].personality_manager.profile} / {LMM_COMPONENTS['mother'].teaching_style}"
    
    neural_info = "Not initialized"
    if LMM_COMPONENTS["neural_substrate"]:
        nn = LMM_COMPONENTS["neural_substrate"]["neural_network"]
        neural_info = f"{len(nn.neurons)} neurons / {len(nn.synapses)} synapses"
    
    return [
        html.Div([
            html.P([html.Strong("Device: "), f"{device}"]),
            html.P([html.Strong("CUDA: "), f"{cuda_info}"]),
            html.P([html.Strong("Modules: "), f"{modules}"]),
            html.P([html.Strong("Mother: "), f"{mother_info}"]),
            html.P([html.Strong("Neural Network: "), f"{neural_info}"]),
            html.P([html.Strong("Session: "), f"{CURRENT_SESSION_ID or 'None'}"]),
        ])
    ]

@app.callback(
    Output('development-status', 'children'),
    Input('main-interval', 'n_intervals')
)
def update_development_status(n_intervals):
    """Update development status display"""
    if not LMM_COMPONENTS["mind"]:
        return html.Div("Mind not initialized")
    
    mind = LMM_COMPONENTS["mind"]
    
    # Get current development status
    age = mind.age
    stage = mind.developmental_stage
    
    # Create stage progress
    stage_ranges = {
        "prenatal": (0.0, 0.1),
        "infant": (0.1, 1.0),
        "child": (1.0, 3.0),
        "adolescent": (3.0, 6.0),
        "adult": (6.0, 10.0)
    }
    
    current_range = stage_ranges.get(stage, (0.0, 1.0))
    stage_progress = ((age - current_range[0]) / (current_range[1] - current_range[0])) * 100
    stage_progress = min(max(stage_progress, 0), 100)  # Ensure between 0-100
    
    # Get module development levels
    module_progress = []
    if mind.modules:
        for module_name, module in mind.modules.items():
            module_progress.append({
                "name": module_name.capitalize(),
                "level": module.development_level * 100  # Convert 0-1 to percentage
            })
    
    return html.Div([
        html.Div([
            html.H3(f"Age: {age:.2f}", style={"margin-bottom": "5px"}),
            html.H4(f"Stage: {stage.capitalize()}", 
                    style={"color": COLORS["stage_colors"].get(stage, COLORS["text_secondary"]), "margin-bottom": "15px"}),
            html.Div([
                html.Label(f"Stage Progress:", style={"margin-bottom": "5px"}),
                dbc.Progress(value=stage_progress, color="info", 
                             style={"height": "10px", "margin-bottom": "20px"}),
            ]),
            html.H5("Module Development:", style={"margin-top": "15px", "margin-bottom": "10px"}),
            html.Div([
                html.Div([
                    html.Label(f"{module['name']}:", style={"margin-bottom": "5px"}),
                    dbc.Progress(value=module['level'], color="primary", 
                                style={"height": "8px", "margin-bottom": "10px"}),
                ]) for module in module_progress
            ]),
        ])
    ])

@app.callback(
    Output('development-graph', 'figure'),
    Input('development-graph-interval', 'n_intervals')
)
def update_development_graph(n_intervals):
    """Update the developmental progression graph"""
    # Get latest session data
    session_data = get_session_data(CURRENT_SESSION_ID)
    
    if not session_data or not session_data.get('interactions'):
        # Return empty figure
        return create_empty_development_figure()
    
    # Extract data from interactions
    interactions = session_data['interactions']
    
    # Prepare data
    timestamps = []
    ages = []
    stages = []
    increments = []
    
    for interaction in interactions:
        timestamps.append(interaction['timestamp'])
        ages.append(interaction['age'])
        stages.append(interaction['developmental_stage'])
        increments.append(interaction['development_increment'])
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add age line
    fig.add_trace(
        go.Scatter(
            x=timestamps, 
            y=ages,
            mode='lines+markers',
            name='Mind Age',
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    # Add development increment bars
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=increments,
            name='Learning Increment',
            marker_color=COLORS["secondary"],
            opacity=0.6
        ),
        secondary_y=True,
    )
    
    # Add stage indicators
    stage_changes = []
    current_stage = None
    for i, stage in enumerate(stages):
        if stage != current_stage:
            stage_changes.append({
                'timestamp': timestamps[i],
                'age': ages[i],
                'stage': stage
            })
            current_stage = stage
    
    for change in stage_changes:
        fig.add_annotation(
            x=change['timestamp'],
            y=change['age'],
            text=change['stage'].capitalize(),
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=COLORS["stage_colors"].get(change['stage'], COLORS["accent"]),
            font=dict(color=COLORS["text"]),
            bgcolor=COLORS["stage_colors"].get(change['stage'], COLORS["accent"]),
            bordercolor=COLORS["stage_colors"].get(change['stage'], COLORS["accent"]),
            borderwidth=2,
            borderpad=4,
            opacity=0.8
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    )
    
    # Update yaxis properties
    fig.update_yaxes(title_text="Mind Age", secondary_y=False)
    fig.update_yaxes(title_text="Learning Increment", secondary_y=True)
    
    return fig

def create_empty_development_figure():
    """Create an empty development graph"""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)"),
        annotations=[dict(
            text="No development data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )]
    )
    return fig

@app.callback(
    Output('conversation-container', 'children'),
    Input('conversation-interval', 'n_intervals')
)
def update_conversation_history(n_intervals):
    """Update the conversation history display"""
    # Get latest session data
    session_data = get_session_data(CURRENT_SESSION_ID)
    
    if not session_data or not session_data.get('interactions'):
        return html.Div("No conversation history available")
    
    # Extract interactions
    interactions = session_data['interactions']
    
    # Create conversation messages
    messages = []
    for i, interaction in enumerate(interactions[-10:]):  # Show only the last 10 interactions
        lmm_input = interaction['lmm_input']
        mother_response = interaction['mother_response']
        
        # Add LMM message
        if lmm_input:
            messages.append(html.Div(lmm_input, className="lmm-message"))
        
        # Add Mother message
        if mother_response:
            messages.append(html.Div(mother_response, className="mother-message"))
    
    return messages

@app.callback(
    Output('neural-activity-graph', 'figure'),
    Input('neural-activity-interval', 'n_intervals')
)
def update_neural_activity(n_intervals):
    """Update the neural activity visualization"""
    if not LMM_COMPONENTS["neural_substrate"]:
        return create_empty_neural_figure()
    
    neural_network = LMM_COMPONENTS["neural_substrate"]["neural_network"]
    
    # Get neuron activations
    neuron_ids = []
    activations = []
    neuron_types = []
    
    for neuron_id, neuron in neural_network.neurons.items():
        neuron_ids.append(str(neuron_id))
        activations.append(neuron.activation)
        
        # Determine neuron type
        if neuron_id in neural_network.input_neurons:
            neuron_types.append("Input")
        elif neuron_id in neural_network.output_neurons:
            neuron_types.append("Output")
        else:
            neuron_types.append("Hidden")
    
    # Get synapse strengths for line thickness
    source_ids = []
    target_ids = []
    weights = []
    
    for synapse_id, synapse in neural_network.synapses.items():
        source_ids.append(str(synapse.source_id))
        target_ids.append(str(synapse.target_id))
        weights.append(synapse.weight)
    
    # Create network visualization
    node_trace = go.Scatter(
        x=[0.2 if t == "Input" else 0.5 if t == "Hidden" else 0.8 for t in neuron_types],
        y=[i * (1.0/len(neuron_ids)) for i in range(len(neuron_ids))],
        mode='markers',
        name='Neurons',
        marker=dict(
            size=[a * 20 + 5 for a in activations],
            color=[COLORS["neural_colors"][i % len(COLORS["neural_colors"])] for i in range(len(neuron_ids))],
            opacity=[a * 0.8 + 0.2 for a in activations],
            line=dict(width=2, color='rgba(255, 255, 255, 0.3)')
        ),
        text=[f"ID: {id}<br>Type: {t}<br>Activation: {a:.3f}" for id, t, a in zip(neuron_ids, neuron_types, activations)],
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=[node_trace])
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        annotations=[
            dict(
                x=0.2,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Input Layer",
                showarrow=False,
                font=dict(size=12, color=COLORS["accent"])
            ),
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Hidden Layers",
                showarrow=False,
                font=dict(size=12, color=COLORS["accent"])
            ),
            dict(
                x=0.8,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Output Layer",
                showarrow=False,
                font=dict(size=12, color=COLORS["accent"])
            )
        ]
    )
    
    return fig

def create_empty_neural_figure():
    """Create an empty neural activity graph"""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text="Neural network not initialized",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )]
    )
    return fig

@app.callback(
    Output('module-cards', 'children'),
    Input('main-interval', 'n_intervals')
)
def update_module_cards(n_intervals):
    """Update the module development cards"""
    if not LMM_COMPONENTS["mind"]:
        return html.Div("Mind not initialized")
    
    mind = LMM_COMPONENTS["mind"]
    
    # Create a card for each module
    cards = []
    
    for i, (module_name, module) in enumerate(mind.modules.items()):
        # Get module state
        module_state = module.get_state()
        development_level = module.development_level * 100  # Convert to percentage
        
        # Create card
        card = dbc.Col([
            dbc.Card([
                dbc.CardHeader(module_name.capitalize()),
                dbc.CardBody([
                    html.H5(f"{development_level:.1f}%", 
                           style={"color": COLORS["accent"], "text-align": "center", "margin-bottom": "15px"}),
                    dbc.Progress(value=development_level, color="primary", 
                                 style={"height": "8px", "margin-bottom": "15px"}),
                    html.Div([
                        html.P(f"Features: {len(module_state.get('features', []))}", 
                               style={"margin-bottom": "5px"}),
                        html.P(f"Patterns: {len(module_state.get('patterns', []))}", 
                               style={"margin-bottom": "5px"}),
                    ])
                ])
            ], className="h-100 module-card")
        ], width=3, style={"margin-bottom": "20px"})
        
        cards.append(card)
    
    # Arrange in rows
    rows = []
    for i in range(0, len(cards), 4):
        rows.append(dbc.Row(cards[i:i+4]))
    
    return html.Div(rows)

@app.callback(
    Output('language-metrics', 'children'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metrics-tabs', 'active_tab')],
)
def update_language_metrics(n_intervals, active_tab):
    """Update language metrics tab"""
    if active_tab != "tab-language" or not LMM_COMPONENTS["mind"]:
        raise PreventUpdate
    
    # Get language module
    mind = LMM_COMPONENTS["mind"]
    language_module = mind.modules.get("language")
    
    if not language_module:
        return html.Div("Language module not available")
    
    # Get state
    language_state = language_module.get_state()
    
    # Get vocabulary
    known_words = language_state.get("known_words", [])
    sentence_patterns = language_state.get("sentence_patterns", [])
    grammar_rules = language_state.get("grammar_rules", [])
    
    # Create visualizations
    
    # Vocabulary growth chart
    vocab_growth = {
        "x": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, mind.age],
        "y": [0, 5, 20, 50, 100, 200, 500, len(known_words)]
    }
    
    vocab_fig = px.line(
        vocab_growth, 
        x="x", 
        y="y", 
        labels={"x": "Age", "y": "Vocabulary Size"},
        title="Vocabulary Growth"
    )
    
    vocab_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=20, t=40, b=40),
    )
    
    # Current vocabulary table
    vocabulary_table = html.Div([
        html.H5("Current Vocabulary", style={"margin-top": "20px", "margin-bottom": "10px"}),
        html.Div([
            html.Div(word, style={
                "display": "inline-block",
                "background-color": "rgba(187, 134, 252, 0.1)",
                "border-radius": "15px",
                "padding": "5px 10px",
                "margin": "5px",
                "font-size": "12px"
            }) for word in sorted(known_words)[:50]  # Show up to 50 words
        ], style={"max-height": "200px", "overflow-y": "auto"})
    ])
    
    # Sentence patterns
    patterns_list = html.Div([
        html.H5("Sentence Patterns", style={"margin-top": "20px", "margin-bottom": "10px"}),
        html.Ul([
            html.Li(pattern) for pattern in sentence_patterns[:10]  # Show up to 10 patterns
        ]) if sentence_patterns else html.P("No sentence patterns learned yet")
    ])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=vocab_fig, style={"height": "300px"})
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H5("Language Development Statistics"),
                    html.Table([
                        html.Tr([
                            html.Td("Development Level:"),
                            html.Td(f"{language_module.development_level:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Vocabulary Size:"),
                            html.Td(f"{len(known_words)} words", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Sentence Patterns:"),
                            html.Td(f"{len(sentence_patterns)}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Grammar Rules:"),
                            html.Td(f"{len(grammar_rules)}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                    ], style={"width": "100%"})
                ], style={"height": "300px"})
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col(vocabulary_table, width=6),
            dbc.Col(patterns_list, width=6),
        ])
    ])

@app.callback(
    Output('memory-metrics', 'children'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metrics-tabs', 'active_tab')],
)
def update_memory_metrics(n_intervals, active_tab):
    """Update memory metrics tab"""
    if active_tab != "tab-memory" or not LMM_COMPONENTS["mind"]:
        raise PreventUpdate
    
    # Get memory module
    mind = LMM_COMPONENTS["mind"]
    memory_module = mind.modules.get("memory")
    
    if not memory_module:
        return html.Div("Memory module not available")
    
    # Get state
    memory_state = memory_module.get_state()
    
    # Get memory metrics
    working_memory_capacity = memory_state.get("working_memory_capacity", 0)
    memory_items = memory_state.get("memory_items", [])
    retention_rate = memory_state.get("retention_rate", 0)
    
    # Create memory type distribution chart
    memory_types = {
        "types": ["Episodic", "Semantic", "Procedural", "Emotional", "Associative"],
        "counts": [
            memory_state.get("episodic_count", 0),
            memory_state.get("semantic_count", 0),
            memory_state.get("procedural_count", 0),
            memory_state.get("emotional_count", 0),
            memory_state.get("associative_count", 0)
        ]
    }
    
    memory_fig = px.bar(
        memory_types,
        x="types",
        y="counts",
        title="Memory Type Distribution",
        color="types",
        color_discrete_sequence=COLORS["module_colors"]
    )
    
    memory_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False
    )
    
    # Recent memories list
    recent_memories = html.Div([
        html.H5("Recent Memories", style={"margin-top": "20px", "margin-bottom": "10px"}),
        html.Div([
            html.Div([
                html.P(memory.get("content", "Unknown memory"), 
                       style={"margin-bottom": "5px"}),
                html.Small([
                    f"Type: {memory.get('type', 'unknown')} | ",
                    f"Strength: {memory.get('strength', 0):.2f} | ",
                    f"Created: {memory.get('created_at', 'unknown')}"
                ], style={"color": COLORS["text_secondary"]})
            ], style={
                "border-left": f"3px solid {COLORS['accent']}",
                "padding-left": "10px",
                "margin-bottom": "15px"
            }) for memory in memory_items[-5:]  # Show 5 most recent memories
        ])
    ])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=memory_fig, style={"height": "300px"})
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H5("Memory System Statistics"),
                    html.Table([
                        html.Tr([
                            html.Td("Development Level:"),
                            html.Td(f"{memory_module.development_level:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Working Memory Capacity:"),
                            html.Td(f"{working_memory_capacity} items", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Total Memories:"),
                            html.Td(f"{len(memory_items)}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Memory Retention Rate:"),
                            html.Td(f"{retention_rate:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                    ], style={"width": "100%"})
                ], style={"height": "300px"})
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col(recent_memories, width=12),
        ])
    ])

@app.callback(
    Output('perception-metrics', 'children'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metrics-tabs', 'active_tab')],
)
def update_perception_metrics(n_intervals, active_tab):
    """Update perception metrics tab"""
    if active_tab != "tab-perception" or not LMM_COMPONENTS["mind"]:
        raise PreventUpdate
    
    # Get session data for perception stats
    session_data = get_session_data(CURRENT_SESSION_ID)
    
    if not session_data or not session_data.get('interactions'):
        return html.Div("No perception data available")
    
    # Get perception module
    mind = LMM_COMPONENTS["mind"]
    perception_module = mind.modules.get("perception")
    
    if not perception_module:
        return html.Div("Perception module not available")
    
    # Get state
    perception_state = perception_module.get_state()
    
    # Extract perception stats from interactions
    interactions = session_data['interactions']
    
    timestamps = []
    novelty_scores = []
    intensity_scores = []
    pattern_counts = []
    
    for interaction in interactions:
        # Parse perception stats JSON
        perception_stats = {}
        if interaction['perception_stats']:
            try:
                perception_stats = json.loads(interaction['perception_stats'])
            except:
                perception_stats = {}
        
        timestamps.append(interaction['timestamp'])
        novelty_scores.append(perception_stats.get('novelty_score', 0))
        intensity_scores.append(perception_stats.get('intensity_score', 0))
        pattern_counts.append(perception_stats.get('pattern_count', 0))
    
    # Create perception metrics chart
    perception_df = {
        "timestamp": timestamps[-20:],  # Last 20 interactions
        "novelty": novelty_scores[-20:],
        "intensity": intensity_scores[-20:],
        "patterns": pattern_counts[-20:]
    }
    
    novelty_fig = px.line(
        perception_df,
        x="timestamp",
        y="novelty",
        title="Perception Novelty Score",
        color_discrete_sequence=[COLORS["accent"]]
    )
    
    novelty_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title=""),
        yaxis=dict(title="Novelty Score")
    )
    
    patterns_fig = px.bar(
        perception_df,
        x="timestamp",
        y="patterns",
        title="Detected Patterns Count",
        color_discrete_sequence=[COLORS["secondary"]]
    )
    
    patterns_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title=""),
        yaxis=dict(title="Pattern Count")
    )
    
    # Pattern types distribution
    pattern_types = perception_state.get("pattern_types", {})
    pattern_type_data = {
        "types": list(pattern_types.keys()) if pattern_types else ["No patterns found"],
        "counts": list(pattern_types.values()) if pattern_types else [0]
    }
    
    pattern_type_fig = px.pie(
        pattern_type_data,
        names="types",
        values="counts",
        title="Pattern Type Distribution",
        color_discrete_sequence=COLORS["module_colors"]
    )
    
    pattern_type_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=novelty_fig, style={"height": "250px"})
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=patterns_fig, style={"height": "250px"})
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Perception Statistics"),
                    html.Table([
                        html.Tr([
                            html.Td("Development Level:"),
                            html.Td(f"{perception_module.development_level:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Pattern Recognition:"),
                            html.Td(f"{perception_state.get('pattern_recognition_level', 0):.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Feature Detection:"),
                            html.Td(f"{perception_state.get('feature_detection_level', 0):.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Average Novelty:"),
                            html.Td(f"{sum(novelty_scores[-10:]) / max(1, len(novelty_scores[-10:])):.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                    ], style={"width": "100%"})
                ])
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=pattern_type_fig, style={"height": "250px"})
            ], width=6),
        ]),
    ])

@app.callback(
    Output('attention-metrics', 'children'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metrics-tabs', 'active_tab')],
)
def update_attention_metrics(n_intervals, active_tab):
    """Update attention metrics tab"""
    if active_tab != "tab-attention" or not LMM_COMPONENTS["mind"]:
        raise PreventUpdate
    
    # Get attention module
    mind = LMM_COMPONENTS["mind"]
    attention_module = mind.modules.get("attention")
    
    if not attention_module:
        return html.Div("Attention module not available")
    
    # Get state
    attention_state = attention_module.get_state()
    
    # Get attention metrics
    focus_duration = attention_state.get("focus_duration", 0)
    attention_capacity = attention_state.get("capacity", 0)
    distractibility = attention_state.get("distractibility", 0)
    
    # Focus history
    focus_history = attention_state.get("focus_history", [])
    
    # If focus history available, create chart
    if focus_history:
        focus_data = {
            "timestamp": [entry.get("timestamp", i) for i, entry in enumerate(focus_history)],
            "focus_level": [entry.get("focus_level", 0) for entry in focus_history],
            "focus_target": [entry.get("focus_target", "unknown") for entry in focus_history]
        }
        
        focus_fig = px.line(
            focus_data,
            x="timestamp",
            y="focus_level",
            title="Attention Focus Level",
            color_discrete_sequence=[COLORS["secondary"]]
        )
        
        focus_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor=COLORS["card_bg"],
            margin=dict(l=40, r=20, t=40, b=40),
        )
    else:
        # Create empty figure
        focus_fig = go.Figure()
        focus_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor=COLORS["card_bg"],
            margin=dict(l=40, r=20, t=40, b=40),
            annotations=[dict(
                text="No focus history data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color=COLORS["text_secondary"])
            )]
        )
    
    # Create attention capacity visualization
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=focus_fig, style={"height": "300px"})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H5("Attention System Statistics"),
                    html.Table([
                        html.Tr([
                            html.Td("Development Level:"),
                            html.Td(f"{attention_module.development_level:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Attention Capacity:"),
                            html.Td(f"{attention_capacity:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Focus Duration:"),
                            html.Td(f"{focus_duration:.2f} seconds", 
                                   style={"color": COLORS["accent"]})
                        ]),
                        html.Tr([
                            html.Td("Distractibility:"),
                            html.Td(f"{distractibility:.2f}", 
                                   style={"color": COLORS["accent"]})
                        ]),
                    ], style={"width": "100%"})
                ], style={"height": "300px"})
            ], width=4),
        ]),
        dbc.Row([
            dbc.Col([
                html.H5("Current Focus", style={"margin-top": "20px", "margin-bottom": "10px"}),
                html.Div([
                    html.Div([
                        html.Strong("Focus Target: "),
                        html.Span(attention_state.get("current_focus", "None"))
                    ]),
                    html.Div([
                        html.Strong("Focus Level: "),
                        html.Span(f"{attention_state.get('current_focus_level', 0):.2f}")
                    ]),
                    html.Div([
                        html.Strong("Duration: "),
                        html.Span(f"{attention_state.get('current_focus_duration', 0):.2f} seconds")
                    ]),
                ], style={
                    "background-color": "rgba(3, 218, 198, 0.1)",
                    "border-radius": "10px",
                    "padding": "15px",
                    "margin-top": "10px"
                })
            ], width=12),
        ]),
    ])

@app.callback(
    Output('learning-metrics', 'children'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metrics-tabs', 'active_tab')],
)
def update_learning_metrics(n_intervals, active_tab):
    """Update learning metrics tab"""
    if active_tab != "tab-learning" or not LMM_COMPONENTS["mind"]:
        raise PreventUpdate
    
    # Get session data
    session_data = get_session_data(CURRENT_SESSION_ID)
    
    if not session_data or not session_data.get('interactions'):
        return html.Div("No learning data available")
    
    # Extract learning increments from interactions
    interactions = session_data['interactions']
    
    timestamps = []
    increments = []
    stages = []
    comprehension_levels = []
    learning_goals = []
    
    for interaction in interactions:
        # Parse interaction details JSON
        details = {}
        if interaction.get('interaction_details'):
            try:
                details = json.loads(interaction['interaction_details'])
            except:
                details = {}
        
        timestamps.append(interaction['timestamp'])
        increments.append(interaction['development_increment'])
        stages.append(interaction['developmental_stage'])
        comprehension_levels.append(details.get('comprehension_level', 'unknown'))
        learning_goals.append(details.get('learning_goal', 'unknown'))
    
    # Create learning increment chart
    learning_df = {
        "timestamp": timestamps[-20:],  # Last 20 interactions
        "increment": increments[-20:],
        "stage": stages[-20:],
        "comprehension": comprehension_levels[-20:],
        "goal": learning_goals[-20:]
    }
    
    increment_fig = px.bar(
        learning_df,
        x="timestamp",
        y="increment",
        color="stage",
        title="Learning Increments by Interaction",
        color_discrete_map=COLORS["stage_colors"]
    )
    
    increment_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title="Stage"
    )
    
    # Create comprehension levels distribution
    comp_counts = {}
    for level in comprehension_levels:
        comp_counts[level] = comp_counts.get(level, 0) + 1
    
    comprehension_df = {
        "level": list(comp_counts.keys()),
        "count": list(comp_counts.values())
    }
    
    comprehension_fig = px.pie(
        comprehension_df,
        names="level",
        values="count",
        title="Comprehension Level Distribution",
        color_discrete_sequence=COLORS["module_colors"]
    )
    
    comprehension_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Recent learning interactions
    recent_learning = html.Div([
        html.H5("Recent Learning Interactions", style={"margin-top": "20px", "margin-bottom": "10px"}),
        html.Div([
            html.Div([
                html.P([
                    html.Strong("Learning Goal: "), 
                    html.Span(learning_goals[-(i+1)] if len(learning_goals) > i else "Unknown")
                ], style={"margin-bottom": "5px"}),
                html.P([
                    html.Strong("Comprehension: "),
                    html.Span(comprehension_levels[-(i+1)] if len(comprehension_levels) > i else "Unknown", 
                             style={"color": COLORS["secondary"]})
                ], style={"margin-bottom": "5px"}),
                html.P([
                    html.Strong("Development Gain: "),
                    html.Span(f"{increments[-(i+1)]:.4f}" if len(increments) > i else "0")
                ], style={"margin-bottom": "0"})
            ], style={
                "border-left": f"3px solid {COLORS['accent']}",
                "padding-left": "10px",
                "margin-bottom": "15px"
            }) for i in range(5)  # Show 5 most recent interactions
        ])
    ])
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=increment_fig, style={"height": "300px"})
            ], width=8),
            dbc.Col([
                dcc.Graph(figure=comprehension_fig, style={"height": "300px"})
            ], width=4),
        ]),
        dbc.Row([
            dbc.Col([
                recent_learning
            ], width=12),
        ]),
    ])

# Run the app
if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(os.path.join("lmm_project", "logs"), exist_ok=True)
    os.makedirs(os.path.join("lmm_project", "storage"), exist_ok=True)
    os.makedirs(os.path.join("lmm_project", "visualization", "output"), exist_ok=True)
    
    # Run the dashboard app
    app.run_server(debug=True, port=8050)
