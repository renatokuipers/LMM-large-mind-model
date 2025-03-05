import os
import time
import json
import logging
import threading
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Literal
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback, Dash, dash_table, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from neural_child import NeuralChild
from mother import Mother, ChildState, MotherResponse
from llm_module import LLMClient, Message
from config import get_config, init_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralChildDashboard")

# Initialize configuration
config = init_config()

# Initialize LLM client
llm_client = LLMClient(base_url=config.system.llm_base_url)

# Check for saved states directory
SAVED_STATES_DIR = Path("./saved_states")
SAVED_STATES_DIR.mkdir(exist_ok=True)

# Define app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Neural Child Dashboard"
server = app.server

# Global state
neural_child = None
mother = None
training_thread = None
training_active = False
training_paused = False
save_interval = 15  # Save every 15 minutes
last_save_time = None

# Helper classes
class TrainingState:
    """Track training state and history"""
    def __init__(self):
        self.metrics_history = {
            "age_days": [],
            "vocabulary_size": [],
            "emotional_development": [],
            "language_acquisition": [],
            "self_awareness": [],
            "memory_formation": [],
            "reasoning_abilities": [],
            "social_understanding": [],
            "belief_formation": [],
            "milestone_average": [],
            "timestamp": [],
        }
        self.conversation_history = []
        self.milestones_reached = False
        self.last_metrics_update = datetime.now()
        self.metrics_update_interval = 5  # seconds
        self.interaction_count = 0
        
        # Network activations history (last 100)
        self.network_activations = {
            "archetypes": [],
            "instincts": [],
            "unconsciousness": [],
            "drives": [],
            "emotions": [],
            "moods": [],
            "attention": [],
            "perception": [],
            "consciousness": [],
            "thoughts": [],
        }
        
    def update_metrics(self, neural_child):
        """Update training metrics from neural child state"""
        current_time = datetime.now()
        if (current_time - self.last_metrics_update).total_seconds() < self.metrics_update_interval:
            return False
            
        # Get milestones
        milestones = neural_child.get_milestone_progress()
        milestone_values = list(milestones.values())
        milestone_avg = sum(milestone_values) / len(milestone_values)
        
        # Update milestone status
        self.milestones_reached = milestone_avg >= 0.75
        
        # Update metrics history
        self.metrics_history["age_days"].append(neural_child.age_days)
        self.metrics_history["vocabulary_size"].append(neural_child.get_vocabulary_size())
        self.metrics_history["emotional_development"].append(milestones["Emotional Development"])
        self.metrics_history["language_acquisition"].append(milestones["Language Acquisition"])
        self.metrics_history["self_awareness"].append(milestones["Self-awareness"])
        self.metrics_history["memory_formation"].append(milestones["Memory Formation"])
        self.metrics_history["reasoning_abilities"].append(milestones["Reasoning Abilities"])
        self.metrics_history["social_understanding"].append(milestones["Social Understanding"])
        self.metrics_history["belief_formation"].append(milestones["Belief Formation"])
        self.metrics_history["milestone_average"].append(milestone_avg)
        self.metrics_history["timestamp"].append(current_time)
        
        # Get network activations
        network_states = neural_child.get_network_states()
        for network_name in self.network_activations.keys():
            if network_name in network_states:
                activation = network_states[network_name].get("activation", 0)
                self.network_activations[network_name].append(activation)
                # Keep only last 100 values
                if len(self.network_activations[network_name]) > 100:
                    self.network_activations[network_name] = self.network_activations[network_name][-100:]
        
        # Update timestamp
        self.last_metrics_update = current_time
        return True

training_state = TrainingState()

# Colors for different metrics
COLORS = {
    "age_days": "#636EFA",
    "vocabulary_size": "#EF553B",
    "emotional_development": "#00CC96",
    "language_acquisition": "#AB63FA",
    "self_awareness": "#FFA15A",
    "memory_formation": "#19D3F3",
    "reasoning_abilities": "#FF6692",
    "social_understanding": "#B6E880",
    "belief_formation": "#FF97FF",
    "milestone_average": "#FECB52",
}

# Training thread function
def training_loop():
    """Main training loop for the neural child"""
    global training_active, training_paused, neural_child, mother, last_save_time
    
    last_save_time = datetime.now()
    
    try:
        while training_active:
            if training_paused:
                time.sleep(0.5)
                continue
                
            # Update neural child state
            neural_child.update_state()
            
            # Update metrics
            training_state.update_metrics(neural_child)
            
            # Every 10 interactions, save state
            if training_state.interaction_count % 10 == 0:
                current_time = datetime.now()
                if last_save_time is None or (current_time - last_save_time).total_seconds() > save_interval * 60:
                    # Save neural child state
                    save_path = neural_child.save_state()
                    last_save_time = current_time
                    logger.info(f"Automatically saved neural child state to {save_path}")
            
            # Generate child utterance
            child_utterance = neural_child.generate_utterance()
            
            # Prepare child state for mother
            child_state = ChildState(
                message=child_utterance,
                apparent_emotion=neural_child.get_apparent_emotion(),
                vocabulary_size=neural_child.get_vocabulary_size(),
                age_days=neural_child.age_days,
                recent_concepts_learned=neural_child.get_recent_concepts_learned(),
                attention_span=neural_child.get_attention_span()
            )
            
            # Get mother's response
            mother_response = mother.respond_to_child(child_state)
            
            # Process mother's response
            neural_child.process_mother_response(mother_response)
            
            # Record conversation
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "child_age": neural_child.age_days,
                "child_message": child_utterance,
                "child_emotion": child_state.apparent_emotion,
                "mother_message": mother_response.verbal.text,
                "mother_emotion": mother_response.emotional.primary_emotion,
            }
            training_state.conversation_history.append(conversation_entry)
            
            # Limit history size
            if len(training_state.conversation_history) > 1000:
                training_state.conversation_history = training_state.conversation_history[-1000:]
            
            # Increment interaction count
            training_state.interaction_count += 1
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        training_active = False

# Layout components

# Header
header = dbc.Row(
    dbc.Col(
        html.Div(
            [
                html.H1("Neural Child Dashboard", className="display-4"),
                html.P(
                    "Visualizing the development of an artificial consciousness",
                    className="lead",
                ),
                html.Hr(className="my-2"),
            ],
            className="p-2 bg-primary rounded-3 text-center",
        ),
        width=12,
    ),
    className="mb-4",
)

# Control panel
control_panel = dbc.Card(
    [
        dbc.CardHeader("Control Panel"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Training Controls"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Initialize", 
                                            id="initialize-btn", 
                                            color="primary",
                                            className="me-1",
                                        ),
                                        dbc.Button(
                                            "Start Training", 
                                            id="start-btn", 
                                            color="success",
                                            className="me-1",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            "Pause", 
                                            id="pause-btn", 
                                            color="warning",
                                            className="me-1",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            "Stop", 
                                            id="stop-btn", 
                                            color="danger",
                                            className="me-1",
                                            disabled=True,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H5("Configuration"),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Development Speed"),
                                        dbc.Input(
                                            id="speed-input",
                                            type="number",
                                            value=10,
                                            min=1,
                                            max=100,
                                            step=1,
                                        ),
                                        dbc.InputGroupText("x"),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Save Interval"),
                                        dbc.Input(
                                            id="save-interval-input",
                                            type="number",
                                            value=15,
                                            min=1,
                                            max=60,
                                            step=1,
                                        ),
                                        dbc.InputGroupText("minutes"),
                                    ],
                                    className="mb-2",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("State Management"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Save State", 
                                            id="save-btn", 
                                            color="info",
                                            className="me-1",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            "Load State", 
                                            id="load-btn", 
                                            color="secondary",
                                            className="me-1",
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dcc.Dropdown(
                                    id="load-state-dropdown",
                                    options=[],
                                    placeholder="Select saved state to load...",
                                    className="mb-2",
                                ),
                            ],
                            width=12,
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id="status-container",
                                    children=[
                                        html.P("Status: Not initialized", id="status-text", className="mb-0"),
                                        html.P("Age: N/A", id="age-text", className="mb-0"),
                                        html.P("Milestone Progress: N/A", id="milestone-text", className="mb-0"),
                                    ],
                                    className="border rounded p-2 bg-dark",
                                ),
                            ],
                            width=12,
                        ),
                    ],
                ),
            ]
        ),
    ],
    className="mb-4",
)

# Chat Panel
chat_panel = dbc.Card(
    [
        dbc.CardHeader("Conversation"),
        dbc.CardBody(
            [
                html.Div(
                    id="conversation-display",
                    className="border rounded p-2 bg-dark overflow-auto mb-3",
                    style={"height": "300px"},
                    children=[
                        html.P(
                            "Training conversation will appear here...",
                            className="text-muted",
                        )
                    ],
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="user-input", 
                            placeholder="Enter message to chat directly (only available after milestones reached)...",
                            disabled=True,
                        ),
                        dbc.Button(
                            "Send", 
                            id="send-btn", 
                            color="primary",
                            disabled=True,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Alert(
                    id="chat-alert",
                    children="Chat will be enabled once all milestones are reached (≥75% average).",
                    color="info",
                    className="mb-0",
                ),
            ]
        ),
    ],
    className="mb-4",
)

# Tabs for different visualizations
development_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Development Milestones"),
                            dcc.Graph(
                                id="milestones-gauge",
                                config={"displayModeBar": False},
                                style={"height": "400px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Milestone Progress Over Time"),
                            dcc.Graph(
                                id="milestones-timeline",
                                config={"displayModeBar": True},
                                style={"height": "400px"},
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Language Development"),
                            dcc.Graph(
                                id="language-graph",
                                config={"displayModeBar": True},
                                style={"height": "300px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Network Activations"),
                            dcc.Graph(
                                id="network-activations",
                                config={"displayModeBar": True},
                                style={"height": "300px"},
                            ),
                        ],
                        width=6,
                    ),
                ],
            ),
        ]
    ),
    className="mt-3",
)

cognition_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Current Thoughts"),
                            html.Div(
                                id="thoughts-container",
                                className="border rounded p-2 bg-dark overflow-auto",
                                style={"height": "200px"},
                                children=[html.P("No thoughts yet...", className="text-muted")],
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Emotional State"),
                            dcc.Graph(
                                id="emotions-chart",
                                config={"displayModeBar": False},
                                style={"height": "200px"},
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Active Memory Items"),
                            html.Div(
                                id="memory-container",
                                className="border rounded p-2 bg-dark overflow-auto",
                                style={"height": "300px"},
                                children=[html.P("No memories yet...", className="text-muted")],
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Personal Identity Development"),
                            html.Div(
                                id="identity-container",
                                className="border rounded p-2 bg-dark overflow-auto",
                                style={"height": "300px"},
                                children=[html.P("No identity formed yet...", className="text-muted")],
                            ),
                        ],
                        width=6,
                    ),
                ],
            ),
        ]
    ),
    className="mt-3",
)

networks_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Neural Networks State"),
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        dbc.Card(
                                            dbc.CardBody(
                                                html.Div(
                                                    id=f"{network}-container",
                                                    className="border rounded p-2 bg-dark overflow-auto",
                                                    style={"height": "150px"},
                                                    children=[html.P(f"No {network} data yet...", className="text-muted")],
                                                )
                                            )
                                        ),
                                        label=network.capitalize(),
                                    )
                                    for network in [
                                        "archetypes", "instincts", "unconsciousness", "drives", 
                                        "emotions", "moods", "attention", "perception", 
                                        "consciousness", "thoughts"
                                    ]
                                ],
                                className="mb-3",
                            ),
                        ],
                        width=12,
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Network Connections"),
                            dcc.Graph(
                                id="network-connections",
                                config={"displayModeBar": True},
                                style={"height": "400px"},
                            ),
                        ],
                        width=12,
                    ),
                ],
            ),
        ]
    ),
    className="mt-3",
)

language_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Vocabulary Growth"),
                            dcc.Graph(
                                id="vocabulary-graph",
                                config={"displayModeBar": True},
                                style={"height": "300px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Grammar Development"),
                            dcc.Graph(
                                id="grammar-graph",
                                config={"displayModeBar": True},
                                style={"height": "300px"},
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Recently Learned Words"),
                            html.Div(
                                id="vocabulary-container",
                                className="border rounded p-2 bg-dark overflow-auto",
                                style={"height": "200px"},
                                children=[html.P("No vocabulary yet...", className="text-muted")],
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Language Stage"),
                            html.Div(
                                id="language-stage-container",
                                className="border rounded p-2 bg-dark overflow-auto",
                                style={"height": "200px"},
                                children=[html.P("No language development yet...", className="text-muted")],
                            ),
                        ],
                        width=6,
                    ),
                ],
            ),
        ]
    ),
    className="mt-3",
)

# Main tabs container
tabs = dbc.Tabs(
    [
        dbc.Tab(development_tab, label="Development Overview"),
        dbc.Tab(cognition_tab, label="Cognition & Memory"),
        dbc.Tab(networks_tab, label="Neural Networks"),
        dbc.Tab(language_tab, label="Language Development"),
    ],
    className="mb-4",
)

# Main layout
app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(control_panel, width=12, md=6),
                dbc.Col(chat_panel, width=12, md=6),
            ],
            className="mb-4",
        ),
        tabs,
        dcc.Interval(
            id="interval-component",
            interval=1000,  # in milliseconds
            n_intervals=0,
        ),
        # Store components for state
        dcc.Store(id="neural-child-initialized", data=False),
        dcc.Store(id="training-active", data=False),
        dcc.Store(id="training-paused", data=False),
        dcc.Store(id="milestones-reached", data=False),
    ],
    fluid=True,
    className="p-4",
)

# Callbacks

@app.callback(
    [Output("load-state-dropdown", "options")],
    [Input("interval-component", "n_intervals")],
)
def update_saved_states_list(n):
    """Update the list of saved states"""
    if not SAVED_STATES_DIR.exists():
        return [[]]
    
    # Get all directories in saved_states
    saved_states = [
        {"label": d.name, "value": str(d)}
        for d in SAVED_STATES_DIR.iterdir()
        if d.is_dir() and (d / "neural_child_state.json").exists()
    ]
    
    # Sort by name (which should include timestamp)
    saved_states.sort(key=lambda x: x["label"], reverse=True)
    
    return [saved_states]

@app.callback(
    [
        Output("initialize-btn", "disabled"),
        Output("start-btn", "disabled"),
        Output("pause-btn", "disabled"),
        Output("stop-btn", "disabled"),
        Output("save-btn", "disabled"),
        Output("neural-child-initialized", "data"),
        Output("training-active", "data"),
        Output("training-paused", "data"),
        Output("status-text", "children"),
        Output("age-text", "children"),
        Output("milestone-text", "children"),
    ],
    [
        Input("initialize-btn", "n_clicks"),
        Input("start-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("save-btn", "n_clicks"),
        Input("load-btn", "n_clicks"),
        Input("interval-component", "n_intervals"),
    ],
    [
        State("speed-input", "value"),
        State("save-interval-input", "value"),
        State("neural-child-initialized", "data"),
        State("training-active", "data"),
        State("training-paused", "data"),
        State("load-state-dropdown", "value"),
    ],
)
def control_training(
    init_clicks, start_clicks, pause_clicks, stop_clicks, save_clicks, load_clicks, 
    n_intervals, speed, save_interval_min, initialized, active, paused, state_to_load
):
    """Control the training process"""
    global neural_child, mother, training_thread, training_active, training_paused, save_interval
    
    # Get which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Return current state if no button was clicked and not initialized
    if not button_id and not initialized:
        return False, True, True, True, True, False, False, False, "Status: Not initialized", "Age: N/A", "Milestone Progress: N/A"
    
    # Initialize neural child
    if button_id == "initialize-btn":
        try:
            neural_child = NeuralChild(development_speed_multiplier=speed)
            mother = Mother(llm_client=llm_client)
            initialized = True
            status_text = "Status: Initialized, ready to start training"
            age_text = f"Age: {neural_child.age_days:.2f} days"
            milestones = neural_child.get_milestone_progress()
            milestone_avg = sum(milestones.values()) / len(milestones)
            milestone_text = f"Milestone Progress: {milestone_avg:.2%}"
            
            # Reset training state
            training_state.__init__()
            training_state.update_metrics(neural_child)
            
            logger.info("Neural child initialized successfully")
            
            # Return updated state with initialization
            return True, False, True, True, False, True, False, False, status_text, age_text, milestone_text
            
        except Exception as e:
            logger.error(f"Error initializing neural child: {str(e)}")
            return False, True, True, True, True, False, False, False, f"Status: Error initializing - {str(e)}", "Age: N/A", "Milestone Progress: N/A"

    # Load state
    elif button_id == "load-btn" and state_to_load:
        try:
            if neural_child is None:
                neural_child = NeuralChild(development_speed_multiplier=speed)
                
            if mother is None:
                mother = Mother(llm_client=llm_client)
                
            # Load the state
            success = neural_child.load_state(Path(state_to_load))
            if success:
                initialized = True
                
                # Reset training state
                training_state.__init__()
                training_state.update_metrics(neural_child)
                
                status_text = "Status: State loaded successfully"
                age_text = f"Age: {neural_child.age_days:.2f} days"
                milestones = neural_child.get_milestone_progress()
                milestone_avg = sum(milestones.values()) / len(milestones)
                milestone_text = f"Milestone Progress: {milestone_avg:.2%}"
                
                logger.info(f"Loaded state from {state_to_load}")
            else:
                status_text = "Status: Error loading state"
                age_text = "Age: N/A"
                milestone_text = "Milestone Progress: N/A"
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            status_text = f"Status: Error loading state - {str(e)}"
            age_text = "Age: N/A"
            milestone_text = "Milestone Progress: N/A"
    
    # Start training
    elif button_id == "start-btn" and initialized and not active:
        if neural_child and mother:
            # Update save interval
            save_interval = save_interval_min
            
            # Start training thread
            training_active = True
            training_paused = False
            training_thread = threading.Thread(target=training_loop)
            training_thread.daemon = True
            training_thread.start()
            
            active = True
            paused = False
            status_text = "Status: Training in progress"
            logger.info("Started training")
        else:
            status_text = "Status: Error starting training - Neural child or mother not initialized"
            
    # Pause training
    elif button_id == "pause-btn" and active:
        if training_active:
            if not training_paused:
                training_paused = True
                paused = True
                status_text = "Status: Training paused"
                logger.info("Paused training")
            else:
                training_paused = False
                paused = False
                status_text = "Status: Training resumed"
                logger.info("Resumed training")
    
    # Stop training
    elif button_id == "stop-btn" and active:
        if training_active:
            training_active = False
            training_paused = False
            if training_thread and training_thread.is_alive():
                # Let the thread finish naturally
                pass
            
            active = False
            paused = False
            status_text = "Status: Training stopped"
            logger.info("Stopped training")
    
    # Save state
    elif button_id == "save-btn" and initialized and neural_child:
        try:
            # Create a directory for the save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = neural_child.save_state()
            
            status_text = f"Status: Saved state to {save_path.name}"
            logger.info(f"Saved state to {save_path}")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            status_text = f"Status: Error saving state - {str(e)}"
    
    # Update status if neural child exists but no button was clicked
    elif initialized and neural_child and n_intervals % 5 == 0:
        age_text = f"Age: {neural_child.age_days:.2f} days"
        milestones = neural_child.get_milestone_progress()
        milestone_avg = sum(milestones.values()) / len(milestones)
        milestone_text = f"Milestone Progress: {milestone_avg:.2%}"
        
        if active:
            if paused:
                status_text = "Status: Training paused"
            else:
                status_text = "Status: Training in progress"
        else:
            status_text = "Status: Ready"
        
        # In this case, we're just updating metrics
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, status_text, age_text, milestone_text
    
    # Default status
    if not initialized:
        status_text = status_text if 'status_text' in locals() else "Status: Not initialized"
        age_text = age_text if 'age_text' in locals() else "Age: N/A"
        milestone_text = milestone_text if 'milestone_text' in locals() else "Milestone Progress: N/A"
    
    # Update button states
    initialize_disabled = initialized and not (active and not paused)
    start_disabled = not initialized or active
    pause_disabled = not active
    stop_disabled = not active
    save_disabled = not initialized
    
    # Set global variables based on UI state
    training_active = active
    training_paused = paused
    
    return initialize_disabled, start_disabled, pause_disabled, stop_disabled, save_disabled, initialized, active, paused, status_text, age_text, milestone_text

@app.callback(
    [
        Output("milestones-gauge", "figure"),
        Output("milestones-timeline", "figure"),
        Output("language-graph", "figure"),
        Output("network-activations", "figure"),
        Output("emotions-chart", "figure"),
        Output("vocabulary-graph", "figure"),
        Output("grammar-graph", "figure"),
        Output("network-connections", "figure"),
    ],
    [Input("interval-component", "n_intervals")],
    [State("neural-child-initialized", "data")],
)
def update_visualizations(n_intervals, initialized):
    """Update all visualization components"""
    if not initialized or neural_child is None:
        # Return empty figures if not initialized
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return [empty_fig] * 8
    
    # Create milestone gauge figure
    milestone_gauge = create_milestone_gauge()
    
    # Create milestone timeline
    milestone_timeline = create_milestone_timeline()
    
    # Create language graph
    language_graph = create_language_graph()
    
    # Create network activations graph
    network_act_graph = create_network_activations_graph()
    
    # Create emotions chart
    emotions_chart = create_emotions_chart()
    
    # Create vocabulary graph
    vocabulary_graph = create_vocabulary_graph()
    
    # Create grammar graph
    grammar_graph = create_grammar_graph()
    
    # Create network connections graph
    network_connections = create_network_connections_graph()
    
    return milestone_gauge, milestone_timeline, language_graph, network_act_graph, emotions_chart, vocabulary_graph, grammar_graph, network_connections

def create_milestone_gauge():
    """Create milestone gauge visualization"""
    if neural_child is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        return empty_fig
        
    milestones = neural_child.get_milestone_progress()
    
    # Create a radial gauge for each milestone
    fig = make_subplots(
        rows=2, 
        cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=list(milestones.keys()) + ["Average"],
    )
    
    milestone_values = list(milestones.values())
    milestone_avg = sum(milestone_values) / len(milestone_values)
    
    # Add milestone gauges
    for i, (milestone, value) in enumerate(milestones.items()):
        row = i // 4 + 1
        col = i % 4 + 1
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": milestone},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1},
                    "bar": {"color": "rgba(50, 168, 82, 0.8)"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 0.25], "color": "rgba(255, 0, 0, 0.3)"},
                        {"range": [0.25, 0.5], "color": "rgba(255, 165, 0, 0.3)"},
                        {"range": [0.5, 0.75], "color": "rgba(255, 255, 0, 0.3)"},
                        {"range": [0.75, 1], "color": "rgba(0, 128, 0, 0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.75,
                    },
                },
                number={"valueformat": ".0%"},
            ),
            row=row,
            col=col,
        )
    
    # Add average gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=milestone_avg,
            title={"text": "Average"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "rgba(50, 168, 82, 0.8)"},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 0.25], "color": "rgba(255, 0, 0, 0.3)"},
                    {"range": [0.25, 0.5], "color": "rgba(255, 165, 0, 0.3)"},
                    {"range": [0.5, 0.75], "color": "rgba(255, 255, 0, 0.3)"},
                    {"range": [0.75, 1], "color": "rgba(0, 128, 0, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.75,
                },
            },
            number={"valueformat": ".0%"},
        ),
        row=2,
        col=4,
    )
    
    fig.update_layout(
        height=400,
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_milestone_timeline():
    """Create milestone timeline visualization"""
    # Check if we have enough data
    if not training_state.metrics_history["age_days"]:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No timeline data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create figure
    fig = go.Figure()
    
    # Add milestone traces
    for metric in [
        "language_acquisition", "emotional_development", "self_awareness",
        "memory_formation", "reasoning_abilities", "social_understanding",
        "belief_formation", "milestone_average"
    ]:
        if not training_state.metrics_history[metric]:
            continue
            
        fig.add_trace(
            go.Scatter(
                x=training_state.metrics_history["age_days"],
                y=training_state.metrics_history[metric],
                mode="lines",
                name=metric.replace("_", " ").title(),
                line=dict(color=COLORS.get(metric, "#ffffff")),
            )
        )
    
    # Add milestone threshold line
    fig.add_shape(
        type="line",
        x0=min(training_state.metrics_history["age_days"]) if training_state.metrics_history["age_days"] else 0,
        y0=0.75,
        x1=max(training_state.metrics_history["age_days"]) if training_state.metrics_history["age_days"] else 1,
        y1=0.75,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(
        title="Milestone Progress Over Time",
        xaxis_title="Age (days)",
        yaxis_title="Progress",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_language_graph():
    """Create language development visualization"""
    # Check if we have enough data
    if not training_state.metrics_history["age_days"]:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No language data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add language acquisition trace
    fig.add_trace(
        go.Scatter(
            x=training_state.metrics_history["age_days"],
            y=training_state.metrics_history["language_acquisition"],
            mode="lines",
            name="Language Acquisition",
            line=dict(color=COLORS["language_acquisition"]),
        ),
        secondary_y=False,
    )
    
    # Add vocabulary size trace
    fig.add_trace(
        go.Scatter(
            x=training_state.metrics_history["age_days"],
            y=training_state.metrics_history["vocabulary_size"],
            mode="lines",
            name="Vocabulary Size",
            line=dict(color=COLORS["vocabulary_size"]),
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Language Development",
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    fig.update_xaxes(title_text="Age (days)")
    fig.update_yaxes(title_text="Language Acquisition", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Vocabulary Size", secondary_y=True)
    
    return fig

def create_network_activations_graph():
    """Create network activations visualization"""
    # Check if we have enough data
    has_data = False
    for activations in training_state.network_activations.values():
        if activations:
            has_data = True
            break
    
    if not has_data:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No network activation data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each network
    for network, activations in training_state.network_activations.items():
        if not activations:
            continue
            
        fig.add_trace(
            go.Scatter(
                y=activations,
                mode="lines",
                name=network.capitalize(),
            )
        )
    
    fig.update_layout(
        title="Recent Network Activations",
        yaxis_title="Activation Level",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_emotions_chart():
    """Create emotions visualization"""
    if neural_child is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        return empty_fig
        
    # Get emotional state
    emotions = neural_child.get_emotional_state()
    
    if not emotions:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No emotional data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=list(emotions.keys()),
            y=list(emotions.values()),
            marker_color=[
                "#FF9999",  # joy
                "#6699FF",  # sadness
                "#FF6666",  # anger
                "#FFCC99",  # fear
                "#99FF99",  # surprise
                "#CC99FF",  # disgust
                "#66CCFF",  # trust
                "#FFFF99",  # anticipation
            ][:len(emotions)],
        )
    )
    
    fig.update_layout(
        title="Current Emotional State",
        yaxis_title="Intensity",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_vocabulary_graph():
    """Create vocabulary growth visualization"""
    # Check if we have enough data
    if not training_state.metrics_history["age_days"]:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No vocabulary data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create figure
    fig = go.Figure()
    
    # Add vocabulary size trace
    fig.add_trace(
        go.Scatter(
            x=training_state.metrics_history["age_days"],
            y=training_state.metrics_history["vocabulary_size"],
            mode="lines",
            name="Vocabulary Size",
            line=dict(color=COLORS["vocabulary_size"]),
        )
    )
    
    fig.update_layout(
        title="Vocabulary Growth Over Time",
        xaxis_title="Age (days)",
        yaxis_title="Words Known",
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_grammar_graph():
    """Create grammar development visualization"""
    if neural_child is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        return empty_fig
        
    # Get language stats
    language_stats = neural_child.get_language_stats()
    
    if not language_stats or "grammar_rules" not in language_stats:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No grammar data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Extract grammar rules and mastery levels
    rules = language_stats.get("grammar_rules", {})
    rule_names = list(rules.keys())
    mastery_levels = [rules[rule]["mastery"] for rule in rule_names]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=rule_names,
            y=mastery_levels,
            marker_color="rgba(50, 171, 96, 0.7)",
        )
    )
    
    fig.update_layout(
        title="Grammar Rules Mastery",
        yaxis_title="Mastery Level",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig

def create_network_connections_graph():
    """Create network connections visualization"""
    if neural_child is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        return empty_fig
        
    # Get network states to extract connections
    network_states = neural_child.get_network_states()
    
    if not network_states:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No network connections data available yet",
            template="plotly_dark",
            font=dict(color="white"),
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return empty_fig
    
    # Create nodes and edges for network graph
    nodes = []
    edges = []
    edge_weights = []
    
    # Add nodes
    for network_name in network_states.keys():
        nodes.append(network_name)
    
    # Add edges
    for source, state in network_states.items():
        connections = state.get("connections", {})
        for target, conn in connections.items():
            edges.append((source, target))
            edge_weights.append(conn.get("strength", 0.5))
    
    # Create figure
    fig = go.Figure()
    
    # Create a complex function to position nodes in a nice arrangement
    def get_node_positions(nodes):
        n = len(nodes)
        positions = {}
        
        # Arrange in a circle
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            positions[node] = (0.8 * np.cos(angle), 0.8 * np.sin(angle))
            
        return positions
    
    node_positions = get_node_positions(nodes)
    
    # Add edges
    for i, (source, target) in enumerate(edges):
        source_pos = node_positions.get(source, (0, 0))
        target_pos = node_positions.get(target, (0, 0))
        weight = edge_weights[i]
        
        # Line thickness based on weight
        line_width = 1 + 5 * weight
        
        # Line color based on weight
        line_color = f"rgba(100, 100, 255, {weight})"
        
        fig.add_trace(
            go.Scatter(
                x=[source_pos[0], target_pos[0]],
                y=[source_pos[1], target_pos[1]],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                hoverinfo="none",
                showlegend=False,
            )
        )
    
    # Add nodes
    for node in nodes:
        pos = node_positions.get(node, (0, 0))
        
        # Get node activation if available
        activation = 0.5
        if node in network_states:
            activation = network_states[node].get("activation", 0.5)
        
        # Node size based on activation
        node_size = 20 + 30 * activation
        
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers+text",
                marker=dict(size=node_size, color=f"rgba(255, 100, 100, {0.3 + 0.7 * activation})"),
                text=[node.capitalize()],
                textposition="middle center",
                name=node.capitalize(),
            )
        )
    
    fig.update_layout(
        title="Neural Network Connections",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        font=dict(color="white"),
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
    )
    
    return fig

@app.callback(
    [
        Output("thoughts-container", "children"),
        Output("memory-container", "children"),
        Output("identity-container", "children"),
        Output("vocabulary-container", "children"),
        Output("language-stage-container", "children"),
    ] + [Output(f"{network}-container", "children") for network in [
        "archetypes", "instincts", "unconsciousness", "drives", 
        "emotions", "moods", "attention", "perception", 
        "consciousness"
    ]],
    [Input("interval-component", "n_intervals")],
    [State("neural-child-initialized", "data")],
)
def update_text_displays(n_intervals, initialized):
    """Update all text display components"""
    if not initialized or neural_child is None:
        # Return placeholders for all outputs
        empty_content = [html.P("No data available yet...", className="text-muted")]
        return [empty_content] * 14
    
    # Update current thoughts display
    thoughts_network = neural_child.networks.get("thoughts")
    thoughts_content = [html.P("No thoughts yet...", className="text-muted")]
    if thoughts_network:
        current_thoughts = thoughts_network.get_current_thoughts()
        if current_thoughts:
            thoughts_content = [
                html.Div([
                    html.P(f"• {thought}", className="mb-1")
                ]) for thought in current_thoughts
            ]
    
    # Update memory display
    recent_memories = neural_child.get_recent_memories(10)
    memory_content = [html.P("No memories yet...", className="text-muted")]
    if recent_memories:
        memory_content = [
            html.Div([
                html.P(f"Memory ID: {mem['id']}", className="font-weight-bold mb-0"),
                html.P(f"Type: {mem['type']}", className="mb-0"),
                html.P(f"Content: {str(mem['content'])[:100]}...", className="mb-0"),
                html.P(f"Tags: {', '.join(mem['tags'])}", className="mb-0"),
                html.Hr(className="my-2")
            ]) for mem in recent_memories
        ]
    
    # Update identity display
    identity_content = [html.P("No identity formed yet...", className="text-muted")]
    longterm_stats = neural_child.get_longterm_memory_stats()
    if longterm_stats:
        identity_items = []
        
        if "belief_count" in longterm_stats:
            identity_items.append(html.P(f"Beliefs: {longterm_stats['belief_count']}", className="mb-0"))
            
        if "value_count" in longterm_stats:
            identity_items.append(html.P(f"Values: {longterm_stats['value_count']}", className="mb-0"))
            
        if "relationship_count" in longterm_stats:
            identity_items.append(html.P(f"Relationships: {longterm_stats['relationship_count']}", className="mb-0"))
            
        if "attribute_count" in longterm_stats:
            identity_items.append(html.P(f"Self-attributes: {longterm_stats['attribute_count']}", className="mb-0"))
            
        if "domain_counts" in longterm_stats:
            identity_items.append(html.H6("Memory Domains:", className="mt-2"))
            for domain, count in longterm_stats["domain_counts"].items():
                identity_items.append(html.P(f"{domain}: {count}", className="mb-0 ms-2"))
                
        if identity_items:
            identity_content = identity_items
    
    # Update vocabulary display
    language_stats = neural_child.get_language_stats()
    vocabulary_content = [html.P("No vocabulary yet...", className="text-muted")]
    if language_stats and "recent_words" in language_stats:
        recent_words = language_stats["recent_words"]
        if recent_words:
            vocabulary_items = []
            for word in recent_words:
                word_stats = language_stats.get("word_stats", {}).get(word, {})
                understanding = word_stats.get("understanding", 0)
                production = word_stats.get("production", 0)
                
                vocabulary_items.append(
                    html.Div([
                        html.P(f"Word: {word}", className="font-weight-bold mb-0"),
                        html.P(f"Understanding: {understanding:.2f}", className="mb-0"),
                        html.P(f"Production: {production:.2f}", className="mb-0"),
                        html.Hr(className="my-1")
                    ])
                )
            vocabulary_content = vocabulary_items
    
    # Update language stage display
    language_stage_content = [html.P("No language development yet...", className="text-muted")]
    if language_stats:
        language_stage_items = []
        
        language_stage_items.append(html.P(f"Stage: {language_stats.get('language_stage', 'Unknown')}", className="mb-1"))
        language_stage_items.append(html.P(f"Vocabulary Size: {language_stats.get('vocabulary_size', 0)}", className="mb-1"))
        language_stage_items.append(html.P(f"Active Vocabulary: {language_stats.get('active_vocabulary', 0)}", className="mb-1"))
        language_stage_items.append(html.P(f"Passive Vocabulary: {language_stats.get('passive_vocabulary', 0)}", className="mb-1"))
        language_stage_items.append(html.P(f"Grammar Complexity: {language_stats.get('grammar_complexity', 0):.2f}", className="mb-1"))
        language_stage_items.append(html.P(f"Comprehension: {language_stats.get('comprehension', 0):.2f}", className="mb-1"))
        language_stage_items.append(html.P(f"Expression: {language_stats.get('expression', 0):.2f}", className="mb-1"))
        
        language_stage_content = language_stage_items
    
    # Update network displays
    network_states = neural_child.get_network_states()
    network_contents = {}
    
    for network in [
        "archetypes", "instincts", "unconsciousness", "drives", 
        "emotions", "moods", "attention", "perception", 
        "consciousness", "thoughts"
    ]:
        network_content = [html.P(f"No {network} data yet...", className="text-muted")]
        
        if network in network_states:
            state = network_states[network]
            network_items = []
            
            # Add activation and confidence
            network_items.append(html.P(f"Activation: {state.get('activation', 0):.2f}", className="mb-0"))
            network_items.append(html.P(f"Confidence: {state.get('confidence', 0):.2f}", className="mb-0"))
            
            # Add network-specific data
            if network == "emotions" and "emotional_state" in state:
                network_items.append(html.H6("Emotional State:", className="mt-2"))
                for emotion, intensity in state["emotional_state"].items():
                    network_items.append(html.P(f"{emotion}: {intensity:.2f}", className="mb-0 ms-2"))
            
            elif network == "drives" and "drive_levels" in state:
                network_items.append(html.H6("Drive Levels:", className="mt-2"))
                for drive, level in state["drive_levels"].items():
                    network_items.append(html.P(f"{drive}: {level:.2f}", className="mb-0 ms-2"))
            
            elif network == "consciousness":
                if "self_awareness" in state:
                    network_items.append(html.P(f"Self-awareness: {state['self_awareness']:.2f}", className="mb-0"))
                if "active_perceptions" in state:
                    network_items.append(html.H6("Active Perceptions:", className="mt-2"))
                    for percept in state["active_perceptions"][:5]:  # Limit to 5
                        network_items.append(html.P(f"• {percept}", className="mb-0 ms-2"))
            
            elif network == "attention" and "focus_objects" in state:
                network_items.append(html.H6("Focus Objects:", className="mt-2"))
                for obj in state["focus_objects"]:
                    network_items.append(html.P(f"• {obj}", className="mb-0 ms-2"))
            
            # Add connections
            if "connections" in state and state["connections"]:
                network_items.append(html.H6("Connections:", className="mt-2"))
                for conn_name, conn in state["connections"].items():
                    network_items.append(html.P(
                        f"→ {conn_name} ({conn['type']}): {conn['strength']:.2f}", 
                        className="mb-0 ms-2"
                    ))
            
            network_content = network_items
            
        network_contents[network] = network_content
    
    return (
        thoughts_content,
        memory_content,
        identity_content,
        vocabulary_content,
        language_stage_content,
        *[network_contents[network] for network in [
            "archetypes", "instincts", "unconsciousness", "drives", 
            "emotions", "moods", "attention", "perception", 
            "consciousness"
        ]]
    )

@app.callback(
    [
        Output("conversation-display", "children"),
        Output("user-input", "disabled"),
        Output("send-btn", "disabled"),
        Output("chat-alert", "children"),
        Output("chat-alert", "color"),
        Output("milestones-reached", "data"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("send-btn", "n_clicks"),
    ],
    [
        State("user-input", "value"),
        State("neural-child-initialized", "data"),
        State("training-active", "data"),
        State("milestones-reached", "data"),
    ],
)
def update_conversation(n_intervals, send_clicks, user_input, initialized, training_active, milestones_reached):
    """Update conversation display and handle user chat"""
    # Set up default returns
    conversation_display = html.P("Training conversation will appear here...", className="text-muted")
    user_input_disabled = True
    send_btn_disabled = True
    chat_alert_text = "Chat will be enabled once all milestones are reached (≥75% average)."
    chat_alert_color = "info"
    
    # Check if neural child is initialized
    if not initialized or neural_child is None:
        return conversation_display, user_input_disabled, send_btn_disabled, chat_alert_text, chat_alert_color, milestones_reached
    
    # Get milestone progress to check if chat should be enabled
    milestones = neural_child.get_milestone_progress()
    milestone_values = list(milestones.values())
    milestone_avg = sum(milestone_values) / len(milestone_values)
    milestone_threshold = 0.75
    
    chat_ready = milestone_avg >= milestone_threshold
    milestones_reached = chat_ready
    
    # Check if we should process a user message
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"] == "send-btn.n_clicks" and send_clicks and user_input and chat_ready:
        # Process user message
        child_response = neural_child.process_chat_message(user_input)
        
        # Add to conversation history
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "child_age": neural_child.age_days,
            "child_message": child_response,
            "child_emotion": neural_child.get_apparent_emotion(),
            "mother_message": user_input,
            "mother_emotion": "neutral",  # User emotion is unknown
        }
        training_state.conversation_history.append(new_entry)
    
    # Update conversation display
    if training_state.conversation_history:
        # Format conversation messages
        conversation_items = []
        
        for entry in training_state.conversation_history[-15:]:  # Show last 15 messages
            # Determine if this is a user-child conversation
            is_user_chat = "mother_message" in entry and entry["mother_message"] and entry.get("child_emotion", None) is not None
            
            if is_user_chat:
                conversation_items.append(
                    html.Div([
                        html.P(
                            f"You: {entry['mother_message']}", 
                            className="mb-1 text-info"
                        ),
                        html.P(
                            f"Neural Child ({entry['child_emotion']}): {entry['child_message']}", 
                            className="mb-1 text-warning"
                        ),
                        html.Hr(className="my-1")
                    ])
                )
            else:
                # Regular mother-child conversation during training
                conversation_items.append(
                    html.Div([
                        html.P(
                            f"Child ({entry['child_emotion']}): {entry['child_message']}",
                            className="mb-1 text-warning"
                        ),
                        html.P(
                            f"Mother ({entry['mother_emotion']}): {entry['mother_message']}",
                            className="mb-1 text-info"
                        ),
                        html.Hr(className="my-1")
                    ])
                )
        
        conversation_display = conversation_items
    
    # Update chat status
    if chat_ready:
        user_input_disabled = False
        send_btn_disabled = False
        chat_alert_text = "Chat is enabled! You can now interact directly with the Neural Child."
        chat_alert_color = "success"
    else:
        remaining = milestone_threshold - milestone_avg
        chat_alert_text = f"Chat will be enabled once all milestones are reached. Current progress: {milestone_avg:.1%}, need {remaining:.1%} more."
        chat_alert_color = "info"
    
    # Disable chat if training is active
    if training_active:
        user_input_disabled = True
        send_btn_disabled = True
        chat_alert_text = "Chat is disabled during training. Pause or stop training to chat."
        chat_alert_color = "warning"
    
    return conversation_display, user_input_disabled, send_btn_disabled, chat_alert_text, chat_alert_color, milestones_reached

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)