"""
Web-based dashboard for the LMM system using Dash.
Provides visualization and interaction with the running system.
"""
import os
import sys
import time
import logging
import threading
from typing import Dict, Any, List, Optional
import datetime
from dotenv import load_dotenv
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from lmm_project.core import get_mind, get_event_bus, get_state_manager, ModuleType, MessageType
from lmm_project.core import StateDict, DevelopmentalStage

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger("lmm_dashboard")

# Initialize the mind and state manager
config_path = os.getenv("CONFIG_PATH", "config.yml")
storage_dir = os.getenv("STORAGE_DIR", "storage")
mind = get_mind(config_path=config_path, storage_dir=storage_dir)
state_manager = get_state_manager(os.path.join(storage_dir, "states"))
event_bus = get_event_bus()

# Initialize Dash app with Bootstrap styling
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "LMM Dashboard"
server = app.server

# State data and caching
system_running = False
state_cache = {}
message_cache = []
module_states = {}

# Maximum points in time-series charts
MAX_POINTS = 1000

# Refresh intervals
FAST_REFRESH = 1000  # 1 second
NORMAL_REFRESH = 5000  # 5 seconds
SLOW_REFRESH = 30000  # 30 seconds

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Large Mind Model Dashboard", className="text-center my-4"),
            html.Hr(),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Control", className="text-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Start", id="start-button", color="success", className="me-2"),
                            dbc.Button("Stop", id="stop-button", color="danger", className="me-2"),
                            dbc.Button("Save State", id="save-state-button", color="info", className="me-2"),
                        ]),
                        dbc.Col([
                            html.Div(id="system-status", className="text-center mt-2"),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Status", className="text-center"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Age", className="text-center"),
                                html.H2(id="age-display", className="text-center display-4")
                            ]),
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4("Stage", className="text-center"),
                                html.H2(id="stage-display", className="text-center display-4")
                            ]),
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4("Runtime", className="text-center"),
                                html.H2(id="runtime-display", className="text-center display-4")
                            ]),
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H4("Milestones", className="text-center"),
                                html.H2(id="milestones-display", className="text-center display-4")
                            ]),
                        ], width=3),
                    ]),
                ]),
            ]),
        ]),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Developmental Progress", className="text-center"),
                dbc.CardBody([
                    dcc.Graph(id="development-chart"),
                ]),
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Homeostasis Status", className="text-center"),
                dbc.CardBody([
                    dcc.Graph(id="homeostasis-chart"),
                ]),
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Messages", className="text-center"),
                dbc.CardBody([
                    dbc.Spinner(html.Div(id="message-log")),
                ]),
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Registered Modules", className="text-center"),
                dbc.CardBody([
                    dbc.Spinner(html.Div(id="module-list")),
                ]),
            ]),
        ], width=6),
    ], className="mb-4"),
    
    dcc.Interval(
        id="fast-interval",
        interval=FAST_REFRESH,
        n_intervals=0
    ),
    dcc.Interval(
        id="normal-interval",
        interval=NORMAL_REFRESH,
        n_intervals=0
    ),
    dcc.Interval(
        id="slow-interval",
        interval=SLOW_REFRESH,
        n_intervals=0
    ),
    
    dbc.Toast(
        id="status-toast",
        header="Status",
        is_open=False,
        dismissable=True,
        duration=4000,
        icon="primary",
        style={"position": "fixed", "top": 10, "right": 10, "width": 350},
    ),
], fluid=True)


# Callback for Start button
@app.callback(
    Output("system-status", "children"),
    Output("start-button", "disabled"),
    Output("stop-button", "disabled"),
    Output("status-toast", "children"),
    Output("status-toast", "header"),
    Output("status-toast", "is_open"),
    Output("status-toast", "icon"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    Input("save-state-button", "n_clicks"),
    State("system-status", "children"),
    prevent_initial_call=True
)
def control_system(start_clicks, stop_clicks, save_clicks, current_status):
    global system_running
    
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if triggered == "start-button" and start_clicks:
        try:
            mind.start()
            system_running = True
            return (
                html.Span("Running", className="text-success"),
                True,
                False,
                "System started successfully",
                "System Started",
                True,
                "success"
            )
        except Exception as e:
            logger.error(f"Error starting system: {str(e)}")
            return (
                current_status,
                False,
                True,
                f"Error starting system: {str(e)}",
                "Error",
                True,
                "danger"
            )
            
    elif triggered == "stop-button" and stop_clicks:
        try:
            mind.stop()
            system_running = False
            return (
                html.Span("Stopped", className="text-danger"),
                False,
                True,
                "System stopped successfully",
                "System Stopped",
                True,
                "warning"
            )
        except Exception as e:
            logger.error(f"Error stopping system: {str(e)}")
            return (
                current_status,
                True,
                False,
                f"Error stopping system: {str(e)}",
                "Error",
                True,
                "danger"
            )
            
    elif triggered == "save-state-button" and save_clicks:
        try:
            filepath = mind.save_state("Manual save from dashboard")
            filename = os.path.basename(filepath)
            return (
                current_status,
                not system_running,
                system_running,
                f"State saved successfully to {filename}",
                "State Saved",
                True,
                "info"
            )
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return (
                current_status,
                not system_running,
                system_running,
                f"Error saving state: {str(e)}",
                "Error",
                True,
                "danger"
            )
    
    # Default - should not reach here
    return (
        current_status,
        not system_running,
        system_running,
        "",
        "",
        False,
        "primary"
    )


# Callback for fast updates (status indicators)
@app.callback(
    Output("age-display", "children"),
    Output("stage-display", "children"),
    Output("runtime-display", "children"),
    Output("milestones-display", "children"),
    Input("fast-interval", "n_intervals")
)
def update_status(n):
    if not system_running:
        return "N/A", "N/A", "N/A", "N/A"
    
    try:
        state = state_manager.get_state()
        
        # Age
        age = state["development"]["age"]
        age_display = f"{age:.2f}"
        
        # Stage
        stage = state["development"]["stage"]
        
        # Runtime
        runtime_seconds = state["system"]["runtime"]
        runtime_display = str(datetime.timedelta(seconds=int(runtime_seconds)))
        
        # Milestones
        milestones = len(state["development"]["milestones_achieved"])
        
        return age_display, stage, runtime_display, str(milestones)
    
    except Exception as e:
        logger.error(f"Error updating status: {str(e)}")
        return "Error", "Error", "Error", "Error"


# Callback for normal updates (charts)
@app.callback(
    Output("development-chart", "figure"),
    Output("homeostasis-chart", "figure"),
    Input("normal-interval", "n_intervals")
)
def update_charts(n):
    # Get current state
    state = state_manager.get_state()
    
    # Update state cache
    timestamp = time.time()
    if "timestamps" not in state_cache:
        state_cache["timestamps"] = []
    if "age" not in state_cache:
        state_cache["age"] = []
    if "energy" not in state_cache:
        state_cache["energy"] = []
    if "arousal" not in state_cache:
        state_cache["arousal"] = []
    if "cognitive_load" not in state_cache:
        state_cache["cognitive_load"] = []
    if "coherence" not in state_cache:
        state_cache["coherence"] = []
    
    # Add current values to cache
    state_cache["timestamps"].append(timestamp)
    state_cache["age"].append(state["development"]["age"])
    state_cache["energy"].append(state["homeostasis"]["energy"])
    state_cache["arousal"].append(state["homeostasis"]["arousal"])
    state_cache["cognitive_load"].append(state["homeostasis"]["cognitive_load"])
    state_cache["coherence"].append(state["homeostasis"]["coherence"])
    
    # Limit cache size
    if len(state_cache["timestamps"]) > MAX_POINTS:
        state_cache["timestamps"] = state_cache["timestamps"][-MAX_POINTS:]
        state_cache["age"] = state_cache["age"][-MAX_POINTS:]
        state_cache["energy"] = state_cache["energy"][-MAX_POINTS:]
        state_cache["arousal"] = state_cache["arousal"][-MAX_POINTS:]
        state_cache["cognitive_load"] = state_cache["cognitive_load"][-MAX_POINTS:]
        state_cache["coherence"] = state_cache["coherence"][-MAX_POINTS:]
    
    # Create development chart
    dev_fig = go.Figure()
    dev_fig.add_trace(go.Scatter(
        x=state_cache["timestamps"],
        y=state_cache["age"],
        mode="lines",
        name="Age"
    ))
    
    # Add stage transition lines
    for stage_value, stage_name in [
        (0.1, "Infant"),
        (1.0, "Child"),
        (3.0, "Adolescent"),
        (6.0, "Adult")
    ]:
        dev_fig.add_shape(
            type="line",
            x0=min(state_cache["timestamps"]),
            y0=stage_value,
            x1=max(state_cache["timestamps"]),
            y1=stage_value,
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dot"),
        )
        
        # Add stage labels
        dev_fig.add_annotation(
            x=max(state_cache["timestamps"]),
            y=stage_value,
            text=stage_name,
            showarrow=False,
            xshift=10,
            bgcolor="rgba(0, 0, 0, 0.5)"
        )
    
    dev_fig.update_layout(
        title="Developmental Age Over Time",
        xaxis_title="Time",
        yaxis_title="Age",
        template="plotly_dark",
        hovermode="closest",
        yaxis=dict(range=[0, max(8.0, max(state_cache["age"]) * 1.1)]),
        xaxis=dict(
            type="date",
            tickformat="%H:%M:%S",
            showgrid=True
        ),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Create homeostasis chart
    home_fig = go.Figure()
    home_fig.add_trace(go.Scatter(
        x=state_cache["timestamps"],
        y=state_cache["energy"],
        mode="lines",
        name="Energy"
    ))
    home_fig.add_trace(go.Scatter(
        x=state_cache["timestamps"],
        y=state_cache["arousal"],
        mode="lines",
        name="Arousal"
    ))
    home_fig.add_trace(go.Scatter(
        x=state_cache["timestamps"],
        y=state_cache["cognitive_load"],
        mode="lines",
        name="Cognitive Load"
    ))
    home_fig.add_trace(go.Scatter(
        x=state_cache["timestamps"],
        y=state_cache["coherence"],
        mode="lines",
        name="Coherence"
    ))
    
    home_fig.update_layout(
        title="Homeostasis Status",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_dark",
        hovermode="closest",
        yaxis=dict(range=[0, 1.1]),
        xaxis=dict(
            type="date",
            tickformat="%H:%M:%S",
            showgrid=True
        ),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return dev_fig, home_fig


# Callback for slow updates (message log and module list)
@app.callback(
    Output("message-log", "children"),
    Output("module-list", "children"),
    Input("slow-interval", "n_intervals")
)
def update_logs_and_modules(n):
    # Get recent messages
    messages = event_bus.get_message_history(limit=10)
    
    # Format message log
    message_items = []
    for msg in messages:
        timestamp = datetime.datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
        
        # Determine badge color based on message type
        color_map = {
            MessageType.PERCEPTION_INPUT: "primary",
            MessageType.ATTENTION_FOCUS: "warning",
            MessageType.MEMORY_STORAGE: "success",
            MessageType.MEMORY_RETRIEVAL: "info",
            MessageType.EMOTION_UPDATE: "danger",
            MessageType.CONSCIOUSNESS_BROADCAST: "light",
            MessageType.SYSTEM_STATUS: "secondary",
        }
        color = color_map.get(msg.message_type, "secondary")
        
        message_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span(f"{timestamp} ", className="text-muted me-2"),
                    dbc.Badge(msg.message_type.name, color=color, className="me-2"),
                    html.Span(f"From: {msg.sender}", className="text-muted me-2"),
                ]),
                html.Div([
                    html.Small(f"Content Type: {msg.content.content_type}"),
                    html.P(str(msg.content.data)[:100] + ("..." if len(str(msg.content.data)) > 100 else "")),
                ]),
            ])
        )
    
    message_log = dbc.ListGroup(
        message_items if message_items else [dbc.ListGroupItem("No messages yet")]
    )
    
    # Get module list
    module_items = []
    for module_id, module_type in mind.module_types.items():
        if module_id in mind.modules:
            module_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        dbc.Badge(module_type.name, color="primary", className="me-2"),
                        html.Span(module_id),
                    ]),
                ])
            )
    
    module_list = dbc.ListGroup(
        module_items if module_items else [dbc.ListGroupItem("No modules registered")]
    )
    
    return message_log, module_list


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
