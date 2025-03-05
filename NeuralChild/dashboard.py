"""
Dashboard interface for the NeuralChild project.

This module provides a web-based dashboard for visualizing the child's mental state,
tracking development, and interacting with the neural child.
"""

import os
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import queue

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from .neural_child import NeuralChild
from .config import DevelopmentalStage
from .core.mother import MotherResponse, ChildInput


class DashboardApp:
    """Dashboard application for the NeuralChild project."""
    
    def __init__(self, neural_child: Optional[NeuralChild] = None):
        """
        Initialize the dashboard.
        
        Args:
            neural_child: Optional NeuralChild instance (created if not provided)
        """
        # Initialize neural child
        self.neural_child = neural_child or NeuralChild()
        
        # Initialize app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
        )
        
        # Configure app
        self.app.title = "NeuralChild: Psychological Mind Simulation"
        
        # Initialize dashboard data
        self.history = {
            'age_days': [],
            'vocabulary_size': [],
            'emotional_stability': [],
            'memory_capacity': [],
            'self_awareness': [],
            'cognitive_complexity': [],
            'language_complexity': [],
            'timestamps': []
        }
        
        # Chat history
        self.chat_history = []
        
        # Development stage descriptions
        self.stage_descriptions = {
            DevelopmentalStage.PRENATAL: "Neural architecture formation",
            DevelopmentalStage.INFANCY: "Babbling, basic recognition",
            DevelopmentalStage.EARLY_CHILDHOOD: "Rapid vocabulary acquisition",
            DevelopmentalStage.MIDDLE_CHILDHOOD: "Grammar emergence",
            DevelopmentalStage.ADOLESCENCE: "Abstract thinking development",
            DevelopmentalStage.EARLY_ADULTHOOD: "Social cognition refinement",
            DevelopmentalStage.MID_ADULTHOOD: "Wisdom development"
        }
        
        # Simulation thread
        self.simulation_thread = None
        self.simulation_queue = queue.Queue()
        self.simulation_active = False
        
        # Set up the app layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        # Header
        header = dbc.Row([
            dbc.Col([
                html.H1("🧠 NeuralChild: Psychological Mind Simulation", className="app-header"),
                html.P("A simulated mind that develops through nurturing interaction", className="lead"),
            ], width=True)
        ], className="header-row")
        
        # Controls section
        controls = dbc.Card([
            dbc.CardHeader("System Controls"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Development Stage"),
                        html.Div(id="current-stage", className="info-text"),
                        html.Div(id="stage-description", className="description-text")
                    ], width=4),
                    dbc.Col([
                        html.H5("Age"),
                        html.Div(id="current-age", className="info-text"),
                        html.H5("Training Time"),
                        html.Div(id="training-time", className="info-text")
                    ], width=4),
                    dbc.Col([
                        html.H5("Development Control"),
                        dbc.Button("Single Interaction", id="single-interaction-btn", color="primary", className="mr-2"),
                        html.Br(),
                        html.Br(),
                        dbc.InputGroup([
                            dbc.Input(id="interactions-input", type="number", min=1, max=1000, step=1, value=10),
                            dbc.InputGroupText("interactions"),
                            dbc.Button("Simulate", id="simulate-btn", color="success")
                        ]),
                        html.Br(),
                        dbc.Button("Stop Simulation", id="stop-simulation-btn", color="danger", disabled=True),
                        html.Div(id="simulation-status", className="status-text")
                    ], width=4)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H5("State Management"),
                        dbc.InputGroup([
                            dbc.Input(id="save-path-input", placeholder="save_state.json", value="neural_child_state.json"),
                            dbc.Button("Save State", id="save-state-btn", color="info")
                        ]),
                        html.Br(),
                        dbc.InputGroup([
                            dbc.Input(id="load-path-input", placeholder="saved_state.json", value="neural_child_state.json"),
                            dbc.Button("Load State", id="load-state-btn", color="warning")
                        ]),
                        html.Div(id="state-status", className="status-text")
                    ], width=6),
                    dbc.Col([
                        html.H5("Simulation Parameters"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Time Acceleration"),
                            dbc.Input(id="time-acceleration-input", type="number", min=1, max=10000, step=1, value=1000)
                        ]),
                        html.Br(),
                        dbc.InputGroup([
                            dbc.InputGroupText("Learning Rate Multiplier"),
                            dbc.Input(id="learning-rate-input", type="number", min=0.1, max=10, step=0.1, value=1.0)
                        ]),
                        html.Div(id="params-status", className="status-text")
                    ], width=6)
                ])
            ])
        ], className="control-card")
        
        # Developmental metrics section
        metrics = dbc.Card([
            dbc.CardHeader("Developmental Metrics"),
            dbc.CardBody([
                dcc.Graph(id="metrics-graph", style={"height": "300px"}),
                dbc.Row([
                    dbc.Col([
                        html.H5("Cognitive Development"),
                        dcc.Graph(id="cognitive-radar", style={"height": "250px"})
                    ], width=6),
                    dbc.Col([
                        html.H5("Component Confidence"),
                        dcc.Graph(id="component-confidence", style={"height": "250px"})
                    ], width=6)
                ])
            ])
        ], className="metrics-card")
        
        # Neural activation visualization
        neural_viz = dbc.Card([
            dbc.CardHeader("Neural Network Activation"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Component Connections"),
                        dcc.Graph(id="neural-connections", style={"height": "350px"})
                    ], width=8),
                    dbc.Col([
                        html.H5("Current Activation"),
                        html.Div(id="activation-values", className="activation-text"),
                        html.Hr(),
                        html.H5("Consciousness State"),
                        html.Div(id="consciousness-state", className="info-text"),
                        html.H5("Attention Focus"),
                        html.Div(id="attention-focus", className="info-text")
                    ], width=4)
                ])
            ])
        ], className="neural-card")
        
        # Interactive chat interface
        chat = dbc.Card([
            dbc.CardHeader("Interaction with Neural Child"),
            dbc.CardBody([
                html.Div(id="chat-container", className="chat-container"),
                dbc.InputGroup([
                    dbc.Input(id="chat-input", placeholder="Enter your message...", disabled=True),
                    dbc.Button("Send", id="send-btn", color="primary", disabled=True)
                ]),
                html.Div(id="chat-status", className="status-text"),
                dbc.Row([
                    dbc.Col([
                        html.H5("Current Emotional State"),
                        html.Div(id="emotional-state", className="info-text")
                    ], width=6),
                    dbc.Col([
                        html.H5("Active Memory Focus"),
                        html.Div(id="memory-focus", className="info-text")
                    ], width=6)
                ])
            ])
        ], className="chat-card")
        
        # Assemble the layout
        self.app.layout = dbc.Container([
            header,
            dbc.Row([
                dbc.Col([controls], width=12)
            ]),
            dbc.Row([
                dbc.Col([metrics], width=12)
            ]),
            dbc.Row([
                dbc.Col([neural_viz], width=12)
            ]),
            dbc.Row([
                dbc.Col([chat], width=12)
            ]),
            # Hidden data store for metrics history
            dcc.Store(id='metrics-history'),
            # Interval for updates
            dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
        ], fluid=True, className="main-container")
    
    def _setup_callbacks(self) -> None:
        """Set up dashboard callbacks."""
        # Update metrics on interval
        @self.app.callback(
            [Output('metrics-history', 'data'),
             Output('current-stage', 'children'),
             Output('stage-description', 'children'),
             Output('current-age', 'children'),
             Output('training-time', 'children')],
            [Input('update-interval', 'n_intervals')],
            [State('metrics-history', 'data')]
        )
        def update_metrics(n_intervals, current_data):
            # Get current metrics
            metrics = self.neural_child.get_developmental_metrics()
            current_time = time.time()
            
            # Initialize data if needed
            if current_data is None:
                current_data = {
                    'age_days': [],
                    'vocabulary_size': [],
                    'emotional_stability': [],
                    'memory_capacity': [],
                    'self_awareness': [],
                    'cognitive_complexity': [],
                    'language_complexity': [],
                    'timestamps': []
                }
            
            # Update the metrics history
            current_data['age_days'].append(metrics['age_days'])
            current_data['vocabulary_size'].append(metrics['vocabulary_size'])
            current_data['emotional_stability'].append(metrics['emotional_stability'])
            current_data['memory_capacity'].append(metrics['memory_capacity'])
            current_data['self_awareness'].append(metrics['self_awareness'])
            current_data['cognitive_complexity'].append(metrics['cognitive_complexity'])
            current_data['language_complexity'].append(metrics['language_complexity'])
            current_data['timestamps'].append(current_time)
            
            # Limit history to 1000 points to avoid performance issues
            if len(current_data['age_days']) > 1000:
                for key in current_data:
                    current_data[key] = current_data[key][-1000:]
            
            # Format developmental stage
            stage = DevelopmentalStage(metrics['developmental_stage'])
            stage_text = f"{stage.name.replace('_', ' ').title()}"
            
            # Get stage description
            description = self.stage_descriptions.get(stage, "")
            
            # Format age
            age_text = f"{metrics['age_days']} days"
            
            # Format training time
            seconds = metrics['training_time_seconds']
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            training_time_text = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            return current_data, stage_text, description, age_text, training_time_text
        
        # Update metrics graph
        @self.app.callback(
            Output('metrics-graph', 'figure'),
            [Input('metrics-history', 'data')]
        )
        def update_metrics_graph(data):
            if not data or len(data['age_days']) < 2:
                # Create empty figure if no data
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.update_layout(
                    title="Development Metrics",
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                return fig
            
            # Create subplots with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add vocabulary size trace (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=data['age_days'],
                    y=data['vocabulary_size'],
                    name="Vocabulary Size",
                    line=dict(color='#FFA15A', width=2)
                ),
                secondary_y=True
            )
            
            # Add other metrics
            fig.add_trace(
                go.Scatter(
                    x=data['age_days'],
                    y=data['emotional_stability'],
                    name="Emotional Stability",
                    line=dict(color='#00CC96', width=2)
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['age_days'],
                    y=data['self_awareness'],
                    name="Self Awareness",
                    line=dict(color='#AB63FA', width=2)
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['age_days'],
                    y=data['language_complexity'],
                    name="Language Complexity",
                    line=dict(color='#EF553B', width=2)
                ),
                secondary_y=False
            )
            
            # Update layout
            fig.update_layout(
                title="Development Metrics",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="<b>Metric Value</b> (0-1 scale)", secondary_y=False)
            fig.update_yaxes(title_text="<b>Vocabulary Size</b>", secondary_y=True)
            
            # Update x-axis
            fig.update_xaxes(title_text="Age (days)")
            
            return fig
        
        # Update cognitive radar chart
        @self.app.callback(
            Output('cognitive-radar', 'figure'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_cognitive_radar(n_intervals):
            # Get current metrics
            metrics = self.neural_child.get_developmental_metrics()
            
            # Define metrics for radar chart
            categories = ['Self Awareness', 'Emotional Stability', 'Language Complexity', 
                        'Cognitive Complexity', 'Social Awareness']
            
            values = [
                metrics['self_awareness'],
                metrics['emotional_stability'],
                metrics['language_complexity'],
                metrics['cognitive_complexity'],
                metrics['social_awareness']
            ]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Cognitive Development',
                line_color='rgb(103, 232, 249)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        # Update component confidence bar chart
        @self.app.callback(
            Output('component-confidence', 'figure'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_component_confidence(n_intervals):
            # Get current metrics
            metrics = self.neural_child.get_developmental_metrics()
            
            # Extract component confidences
            components = list(metrics['component_confidence'].keys())
            confidences = [metrics['component_confidence'].get(comp, 0) for comp in components]
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=components,
                y=confidences,
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                text=[f"{conf:.2f}" for conf in confidences],
                textposition='auto'
            ))
            
            fig.update_layout(
                yaxis=dict(range=[0, 1]),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        # Update neural connections network graph
        @self.app.callback(
            Output('neural-connections', 'figure'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_neural_connections(n_intervals):
            # Create networkx graph
            G = nx.DiGraph()
            
            # Add nodes for each component
            component_states = self.neural_child.get_component_states()
            components = list(component_states.keys())
            
            for component in components:
                G.add_node(component)
            
            # Add connections between components
            connections = [
                ("Emotion", "Language"),
                ("Language", "Memory"),
                ("Emotion", "Memory"),
                ("Memory", "Consciousness"),
                ("Language", "Consciousness"),
                ("Emotion", "Consciousness")
            ]
            
            for source, target in connections:
                if source in components and target in components:
                    G.add_edge(source, target)
            
            # Get node positions
            pos = {
                "Emotion": [0, 1],
                "Language": [2, 1],
                "Memory": [1, 0],
                "Consciousness": [1, -1],
                # Add positions for additional components as needed
            }
            
            # Get node sizes based on activation
            node_sizes = []
            for node in G.nodes():
                activation = component_states.get(node, {}).get('activation', 0.3)
                # Scale activation to size (30-80)
                size = 30 + (activation * 50)
                node_sizes.append(size)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Create edge trace
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=1, color='#FFFFFF'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                text=list(G.nodes()),
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=node_sizes,
                    color=[component_states.get(node, {}).get('activation', 0.3) for node in G.nodes()],
                    colorbar=dict(
                        thickness=15,
                        title='Activation',
                        xanchor='left'
                        # Remove the titleside property as it's not supported
                    ),
                    line=dict(width=2)
                ),
                textposition="top center"
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace])
            
            # Update layout
            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        # Update activation values
        @self.app.callback(
            Output('activation-values', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_activation_values(n_intervals):
            # Get component states
            component_states = self.neural_child.get_component_states()
            
            # Create activation bars for each component
            activation_bars = []
            for component, state in component_states.items():
                activation = state.get('activation', 0)
                confidence = state.get('confidence', 0)
                
                # Create activation bar
                bar = html.Div([
                    html.Div(f"{component}: {activation:.2f}", className="activation-label"),
                    html.Div(className="activation-bar", style={
                        "width": f"{activation * 100}%",
                        "background-color": f"rgba(103, 232, 249, {confidence})"
                    })
                ], className="activation-container")
                
                activation_bars.append(bar)
            
            return activation_bars
        
        # Update consciousness state and attention focus
        @self.app.callback(
            [Output('consciousness-state', 'children'),
             Output('attention-focus', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_consciousness_info(n_intervals):
            # Get component states
            component_states = self.neural_child.internal_state.component_states
            
            # Get consciousness state
            consciousness_state = "Unknown"
            attention_focus = "Unfocused"
            
            if "Consciousness" in component_states:
                consciousness_info = component_states["Consciousness"]
                consciousness_state = consciousness_info.get('consciousness_state', 'Unknown')
                attention_focus = consciousness_info.get('attention_focus', 'Unfocused')
            
            return consciousness_state, attention_focus
        
        # Update emotional state and memory focus
        @self.app.callback(
            [Output('emotional-state', 'children'),
             Output('memory-focus', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_emotion_memory_info(n_intervals):
            # Get component states
            component_states = self.neural_child.internal_state.component_states
            
            # Get emotional state
            emotional_state = "Neutral"
            if "Emotion" in component_states:
                emotion_info = component_states["Emotion"]
                primary_emotion = emotion_info.get('primary_emotion')
                if primary_emotion:
                    intensity = emotion_info.get('emotional_state', {}).get('categories', {}).get(primary_emotion, 0.5)
                    emotional_state = f"{primary_emotion.capitalize()} ({intensity:.2f})"
            
            # Get memory focus
            memory_focus = "None"
            if "Memory" in component_states:
                memory_info = component_states["Memory"]
                focus = memory_info.get('focus')
                if focus:
                    memory_content = focus.get('content', {})
                    if 'text' in memory_content:
                        memory_focus = memory_content['text'][:50] + "..." if len(memory_content['text']) > 50 else memory_content['text']
                    else:
                        memory_focus = str(memory_content)[:50] + "..." if len(str(memory_content)) > 50 else str(memory_content)
            
            return emotional_state, memory_focus
        
        # Update chat display
        @self.app.callback(
            Output('chat-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_chat_display(n_intervals):
            # Create chat messages
            chat_elements = []
            
            for msg in self.chat_history:
                if msg['sender'] == 'mother':
                    # Mother message
                    chat_elements.append(html.Div([
                        html.Div("👩 Mother:", className="chat-sender"),
                        html.Div(msg['content'], className="chat-message mother-message")
                    ], className="chat-row"))
                else:
                    # Child message
                    chat_elements.append(html.Div([
                        html.Div("👶 Child:", className="chat-sender"),
                        html.Div(msg['content'], className="chat-message child-message")
                    ], className="chat-row"))
            
            return chat_elements
        
        # Enable/disable chat based on development stage
        @self.app.callback(
            [Output('chat-input', 'disabled'),
             Output('send-btn', 'disabled'),
             Output('chat-status', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_chat_status(n_intervals):
            # Get current metrics
            metrics = self.neural_child.get_developmental_metrics()
            
            # Determine if chat should be enabled
            # Enable chat for middle childhood or higher
            stage = DevelopmentalStage(metrics['developmental_stage'])
            chat_enabled = stage.value >= DevelopmentalStage.MIDDLE_CHILDHOOD.value
            
            if chat_enabled:
                return False, False, ""
            else:
                return True, True, "Chat interface will be enabled once child reaches Middle Childhood stage"
        
        # Single interaction button
        @self.app.callback(
            Output('single-interaction-btn', 'disabled'),
            [Input('single-interaction-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_single_interaction(n_clicks):
            if n_clicks is None:
                return False
            
            # Perform single interaction
            interaction_result = self.neural_child.interact_with_mother()
            
            # Extract child and mother messages
            child_input = interaction_result.get('child_input', {})
            mother_response = interaction_result.get('mother_response', {})
            
            # Add to chat history
            child_content = child_input.get('content', '')
            mother_content = mother_response.get('verbal_response', '')
            
            if child_content:
                self.chat_history.append({
                    'sender': 'child',
                    'content': child_content,
                    'timestamp': time.time()
                })
            
            if mother_content:
                self.chat_history.append({
                    'sender': 'mother',
                    'content': mother_content,
                    'timestamp': time.time()
                })
            
            # Limit chat history
            if len(self.chat_history) > 50:
                self.chat_history = self.chat_history[-50:]
            
            return False
        
        # Start simulation
        @self.app.callback(
            [Output('simulate-btn', 'disabled'),
             Output('stop-simulation-btn', 'disabled'),
             Output('simulation-status', 'children')],
            [Input('simulate-btn', 'n_clicks'),
             Input('stop-simulation-btn', 'n_clicks')],
            [State('interactions-input', 'value')],
            prevent_initial_call=True
        )
        def handle_simulation(start_clicks, stop_clicks, interactions):
            triggered_id = ctx.triggered_id
            
            if triggered_id == 'simulate-btn' and start_clicks:
                # Start simulation
                if self.simulation_thread is None or not self.simulation_thread.is_alive():
                    self.simulation_active = True
                    self.simulation_thread = threading.Thread(
                        target=self._run_simulation, 
                        args=(int(interactions),)
                    )
                    self.simulation_thread.daemon = True
                    self.simulation_thread.start()
                    
                    return True, False, "Simulation running..."
            
            elif triggered_id == 'stop-simulation-btn' and stop_clicks:
                # Stop simulation
                self.simulation_active = False
                return False, True, "Simulation stopped."
            
            # Check if simulation is still running
            if self.simulation_thread and self.simulation_thread.is_alive():
                return True, False, "Simulation running..."
            else:
                return False, True, ""
        
        # Send chat message
        @self.app.callback(
            Output('chat-input', 'value'),
            [Input('send-btn', 'n_clicks')],
            [State('chat-input', 'value')],
            prevent_initial_call=True
        )
        def send_chat_message(n_clicks, chat_input):
            if n_clicks is None or not chat_input:
                return ""
            
            # Add message to chat history
            self.chat_history.append({
                'sender': 'mother',
                'content': chat_input,
                'timestamp': time.time()
            })
            
            # Process interaction with manual mother input
            interaction_result = self.neural_child.interact_with_mother(chat_input)
            
            # Extract child message
            child_input = interaction_result.get('child_input', {})
            child_content = child_input.get('content', '')
            
            if child_content:
                self.chat_history.append({
                    'sender': 'child',
                    'content': child_content,
                    'timestamp': time.time()
                })
            
            # Limit chat history
            if len(self.chat_history) > 50:
                self.chat_history = self.chat_history[-50:]
            
            return ""
        
        # Save state
        @self.app.callback(
            Output('state-status', 'children'),
            [Input('save-state-btn', 'n_clicks'),
             Input('load-state-btn', 'n_clicks')],
            [State('save-path-input', 'value'),
             State('load-path-input', 'value')],
            prevent_initial_call=True
        )
        def handle_state_management(save_clicks, load_clicks, save_path, load_path):
            triggered_id = ctx.triggered_id
            
            if triggered_id == 'save-state-btn' and save_clicks:
                # Save state
                if save_path:
                    success = self.neural_child.save_state(save_path)
                    if success:
                        return f"State saved to {save_path}"
                    else:
                        return f"Error saving state to {save_path}"
                else:
                    return "Please enter a save path"
            
            elif triggered_id == 'load-state-btn' and load_clicks:
                # Load state
                if load_path:
                    success = self.neural_child.load_state(load_path)
                    if success:
                        return f"State loaded from {load_path}"
                    else:
                        return f"Error loading state from {load_path}"
                else:
                    return "Please enter a load path"
            
            return ""
        
        # Update simulation parameters
        @self.app.callback(
            Output('params-status', 'children'),
            [Input('time-acceleration-input', 'value'),
             Input('learning-rate-input', 'value')],
            prevent_initial_call=True
        )
        def update_simulation_parameters(time_acceleration, learning_rate):
            # Update parameters
            if self.neural_child.development_params:
                try:
                    self.neural_child.development_params["time_acceleration"] = time_acceleration
                    
                    # Apply learning rate to all components evenly if it's a dictionary
                    if isinstance(self.neural_child.development_params["learning_rate_multiplier"], dict):
                        for key in self.neural_child.development_params["learning_rate_multiplier"]:
                            self.neural_child.development_params["learning_rate_multiplier"][key] = learning_rate
                    
                    return "Parameters updated"
                except Exception as e:
                    return f"Error updating parameters: {str(e)}"
            
            return ""
    
    def _run_simulation(self, interactions: int) -> None:
        """
        Run a simulation for a specified number of interactions.
        
        Args:
            interactions: Number of interactions to simulate
        """
        try:
            for i in range(interactions):
                if not self.simulation_active:
                    break
                
                # Perform interaction
                interaction_result = self.neural_child.interact_with_mother()
                
                # Extract child and mother messages
                child_input = interaction_result.get('child_input', {})
                mother_response = interaction_result.get('mother_response', {})
                
                # Add to chat history
                child_content = child_input.get('content', '')
                mother_content = mother_response.get('verbal_response', '')
                
                if child_content:
                    self.chat_history.append({
                        'sender': 'child',
                        'content': child_content,
                        'timestamp': time.time()
                    })
                
                if mother_content:
                    self.chat_history.append({
                        'sender': 'mother',
                        'content': mother_content,
                        'timestamp': time.time()
                    })
                
                # Limit chat history
                if len(self.chat_history) > 50:
                    self.chat_history = self.chat_history[-50:]
                
                # Update status in queue
                self.simulation_queue.put({
                    'current': i + 1,
                    'total': interactions
                })
                
                # Sleep to avoid overloading the system
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Simulation error: {e}")
        
        finally:
            # Mark simulation as complete
            self.simulation_active = False
    
    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run on
        """
        self.app.run_server(debug=debug, port=port)


def main():
    """Run the dashboard application."""
    dashboard = DashboardApp()
    dashboard.run_server(debug=True)


if __name__ == "__main__":
    main()