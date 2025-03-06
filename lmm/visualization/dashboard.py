"""
Visualization dashboard for the Large Mind Model (LMM).

This module implements a web-based dashboard for visualizing the
development and state of the LMM, using Dash and Plotly.
"""
import os
import time
import threading
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import networkx as nx

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.development.stages import DevelopmentalStage

logger = get_logger("lmm.visualization.dashboard")

class DevelopmentDashboard:
    """
    Dashboard for visualizing the development of the LMM.
    
    This class implements a web-based dashboard using Dash and Plotly
    to visualize the developmental state, memory, emotional state,
    and other aspects of the LMM system.
    """
    
    def __init__(self, lmm_instance=None, port: int = 8050):
        """
        Initialize the development dashboard.
        
        Args:
            lmm_instance: Instance of the LMM
            port: Port to run the dashboard on
        """
        self.lmm = lmm_instance
        self.port = port
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, 
                             title="LMM Development Dashboard",
                             meta_tags=[{"name": "viewport", 
                                        "content": "width=device-width, initial-scale=1"}])
        
        # Store for historical data
        self.metrics_history = []
        self.emotion_history = []
        self.memory_history = []
        self.conversation_history = []
        self.memory_network_data = {"nodes": [], "edges": []}
        self.working_memory_history = []
        self.consolidation_history = []
        self.retrieval_stats = {"counts": [], "scores": [], "timestamps": []}
        
        # Set up layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Server for background running
        self.server = self.app.server
        self.thread = None
        self.running = False
        
        logger.info("Initialized LMM Development Dashboard")
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1("Large Mind Model - Development Dashboard"),
                        html.P("Monitoring the development and state of the LMM system"),
                    ],
                    className="header",
                ),
                
                # Main content area
                html.Div(
                    [
                        # Left column - development metrics
                        html.Div(
                            [
                                html.H2("Developmental Status"),
                                html.Div([
                                    html.Strong("Current Stage: "),
                                    html.Span(id="current-stage"),
                                ]),
                                html.Div([
                                    html.Strong("Progress: "),
                                    html.Span(id="stage-progress"),
                                ]),
                                html.H3("Development Metrics"),
                                dcc.Graph(id="metrics-graph"),
                                
                                html.H3("Advanced Learning Metrics"),
                                dcc.Graph(id="advanced-learning-graph"),
                                
                                # Refresh interval
                                dcc.Interval(
                                    id="metrics-interval",
                                    interval=5000,  # 5 seconds
                                    n_intervals=0,
                                ),
                            ],
                            className="column",
                            style={"width": "32%"},
                        ),
                        
                        # Middle column - emotional state and memory
                        html.Div(
                            [
                                html.H2("Emotional State"),
                                html.Div(id="emotional-state"),
                                dcc.Graph(id="emotion-graph"),
                                dcc.Interval(
                                    id="emotion-interval",
                                    interval=5000,
                                    n_intervals=0,
                                ),
                                
                                html.H2("Memory System"),
                                html.Div(id="memory-stats"),
                                dcc.Graph(id="memory-graph"),
                                dcc.Interval(
                                    id="memory-interval",
                                    interval=5000,
                                    n_intervals=0,
                                ),
                                
                                html.H3("Memory Consolidation"),
                                dcc.Graph(id="consolidation-graph"),
                            ],
                            className="column",
                            style={"width": "32%"},
                        ),
                        
                        # Right column - memory network and conversation
                        html.Div(
                            [
                                html.H2("Memory Network"),
                                dcc.Graph(id="memory-network-graph"),
                                
                                html.H3("Working Memory"),
                                html.Div(id="working-memory"),
                                dcc.Graph(id="working-memory-graph"),
                                
                                html.H3("Memory Retrieval Stats"),
                                dcc.Graph(id="retrieval-stats-graph"),
                                
                                dcc.Interval(
                                    id="memory-network-interval",
                                    interval=10000,  # 10 seconds
                                    n_intervals=0,
                                ),
                                
                                html.H2("Conversation History"),
                                html.Div(id="conversation-history", style={"maxHeight": "300px", "overflow": "auto"}),
                                dcc.Interval(
                                    id="conversation-interval",
                                    interval=5000,
                                    n_intervals=0,
                                ),
                            ],
                            className="column",
                            style={"width": "32%"},
                        ),
                    ],
                    className="row",
                ),
            ],
            className="container",
        )
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [Output("current-stage", "children"),
             Output("stage-progress", "children"),
             Output("metrics-graph", "figure"),
             Output("advanced-learning-graph", "figure")],
            [Input("metrics-interval", "n_intervals")]
        )
        def update_metrics(n):
            if not self.lmm:
                return "Unknown", "0%", {}, {}
            
            # Get development status
            status = self.lmm.get_development_status()
            
            # Store in history
            self.metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "stage": status.get("current_stage", "unknown"),
                "progress": status.get("progress_in_stage", 0),
                **status.get("brain_development", {})
            })
            
            # Limit history size
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # Create dataframe for metrics graph
            if self.metrics_history:
                df = pd.DataFrame(self.metrics_history)
                
                # Format timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Create metrics graph
                metrics_fig = go.Figure()
                
                for metric in ["language_capacity", "emotional_awareness", 
                               "social_understanding", "cognitive_capability", 
                               "self_awareness"]:
                    if metric in df.columns:
                        metrics_fig.add_trace(
                            go.Scatter(
                                x=df["timestamp"],
                                y=df[metric],
                                mode="lines+markers",
                                name=metric.replace("_", " ").title()
                            )
                        )
                
                metrics_fig.update_layout(
                    title="Development Metrics Over Time",
                    xaxis_title="Time",
                    yaxis_title="Metric Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                # Get learning metrics
                learning_metrics = status.get("learning_metrics", {})
                
                # Create advanced learning graph
                adv_learning_fig = go.Figure()
                
                # Add cognitive load bar
                adv_learning_fig.add_trace(
                    go.Bar(
                        x=["Cognitive Load"],
                        y=[learning_metrics.get("cognitive_load", 0)],
                        name="Cognitive Load"
                    )
                )
                
                # Add cognitive capacity bar
                adv_learning_fig.add_trace(
                    go.Bar(
                        x=["Cognitive Capacity"],
                        y=[learning_metrics.get("cognitive_capacity", 0)],
                        name="Cognitive Capacity"
                    )
                )
                
                # Add attention focus indicator
                attention_focus = learning_metrics.get("current_attention_focus", "unknown")
                adv_learning_fig.add_annotation(
                    x=0.5,
                    y=0.9,
                    xref="paper",
                    yref="paper",
                    text=f"Current Attention Focus: {attention_focus}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                adv_learning_fig.update_layout(
                    title="Advanced Learning Metrics",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                
            else:
                metrics_fig = go.Figure()
                adv_learning_fig = go.Figure()
            
            return (
                status.get("current_stage", "Unknown"),
                f"{status.get('progress_in_stage', 0) * 100:.1f}%",
                metrics_fig,
                adv_learning_fig
            )
        
        @self.app.callback(
            [Output("emotional-state", "children"),
             Output("emotion-graph", "figure")],
            [Input("emotion-interval", "n_intervals")]
        )
        def update_emotions(n):
            if not self.lmm:
                return html.Div("No data available"), {}
            
            # Get emotional module status
            try:
                modules_status = self.lmm.get_mind_modules_status()
                emotional_status = modules_status.get("emotional", {})
                
                # Get emotional state
                emotional_state = emotional_status.get("current_state", {})
                
                # Store in history
                self.emotion_history.append({
                    "timestamp": datetime.now().isoformat(),
                    **emotional_state
                })
                
                # Limit history size
                if len(self.emotion_history) > 100:
                    self.emotion_history = self.emotion_history[-100:]
                
                # Create emotion status display
                emotion_status_items = []
                for emotion, intensity in emotional_state.items():
                    if intensity > 0.1:  # Only show significant emotions
                        emotion_status_items.append(
                            html.Div([
                                html.Strong(f"{emotion.title()}: "),
                                html.Span(f"{intensity:.2f}")
                            ])
                        )
                
                if not emotion_status_items:
                    emotion_status_items.append(html.Div("No significant emotions"))
                
                emotion_status_display = html.Div(emotion_status_items)
                
                # Create emotion graph
                if self.emotion_history:
                    df = pd.DataFrame(self.emotion_history)
                    
                    # Format timestamp
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    emotion_fig = go.Figure()
                    
                    # Add lines for each emotion
                    for emotion in emotional_state.keys():
                        if emotion in df.columns:
                            emotion_fig.add_trace(
                                go.Scatter(
                                    x=df["timestamp"],
                                    y=df[emotion],
                                    mode="lines",
                                    name=emotion.title()
                                )
                            )
                    
                    emotion_fig.update_layout(
                        title="Emotional State Over Time",
                        xaxis_title="Time",
                        yaxis_title="Intensity",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                else:
                    emotion_fig = go.Figure()
                
                return emotion_status_display, emotion_fig
                
            except Exception as e:
                logger.error(f"Error updating emotion display: {e}")
                return html.Div(f"Error: {e}"), go.Figure()
        
        @self.app.callback(
            [Output("memory-stats", "children"),
             Output("memory-graph", "figure"),
             Output("consolidation-graph", "figure")],
            [Input("memory-interval", "n_intervals")]
        )
        def update_memory_stats(n):
            if not self.lmm:
                return html.Div("No data available"), {}, {}
            
            try:
                # Get memory status
                memory_status = self.lmm.get_memory_status()
                memory_stats = memory_status.get("memory_stats", {})
                
                # Store in history with timestamp
                self.memory_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "total_memories": memory_stats.get("total_memories", 0),
                    "memory_types": memory_stats.get("memory_types", {}),
                    "memory_strength": memory_stats.get("strength_distribution", {})
                })
                
                # Store consolidation stats
                consolidation_timestamp = datetime.now().isoformat()
                self.consolidation_history.append({
                    "timestamp": consolidation_timestamp,
                    "strength_weak": memory_stats.get("strength_distribution", {}).get("weak", 0),
                    "strength_moderate": memory_stats.get("strength_distribution", {}).get("moderate", 0),
                    "strength_strong": memory_stats.get("strength_distribution", {}).get("strong", 0),
                    "strength_permanent": memory_stats.get("strength_distribution", {}).get("permanent", 0),
                })
                
                # Limit history size
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                if len(self.consolidation_history) > 100:
                    self.consolidation_history = self.consolidation_history[-100:]
                
                # Create memory stats display
                memory_types_counts = memory_stats.get("memory_types", {})
                memory_strength_counts = memory_stats.get("strength_distribution", {})
                
                memory_stats_items = [
                    html.Div([
                        html.Strong("Total Memories: "),
                        html.Span(f"{memory_stats.get('total_memories', 0)}")
                    ]),
                    html.Div([
                        html.Strong("Working Memory: "),
                        html.Span(f"{len(memory_status.get('working_memory', []))}/{memory_stats.get('working_memory_capacity', 0)}")
                    ]),
                    html.H4("Memory Types"),
                    html.Div([
                        html.Div([
                            html.Strong(f"{memory_type.title()}: "),
                            html.Span(f"{count}")
                        ]) for memory_type, count in memory_types_counts.items()
                    ]),
                    html.H4("Memory Strength Distribution"),
                    html.Div([
                        html.Div([
                            html.Strong(f"{strength.title()}: "),
                            html.Span(f"{count}")
                        ]) for strength, count in memory_strength_counts.items()
                    ])
                ]
                
                memory_stats_display = html.Div(memory_stats_items)
                
                # Create memory graph
                memory_fig = go.Figure()
                
                # Add pie chart for memory types
                if memory_types_counts:
                    labels = list(memory_types_counts.keys())
                    values = list(memory_types_counts.values())
                    
                    memory_fig.add_trace(
                        go.Pie(
                            labels=labels,
                            values=values,
                            name="Memory Types",
                            title="Memory Types Distribution"
                        )
                    )
                    
                    memory_fig.update_layout(
                        title="Memory Types Distribution",
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                
                # Create consolidation graph (stacked area chart)
                consolidation_fig = go.Figure()
                
                if self.consolidation_history:
                    df = pd.DataFrame(self.consolidation_history)
                    
                    # Format timestamp
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    consolidation_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["strength_weak"],
                            mode="lines",
                            name="Weak",
                            stackgroup="one",
                            line=dict(width=1, color="rgba(255, 0, 0, 0.8)")
                        )
                    )
                    
                    consolidation_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["strength_moderate"],
                            mode="lines",
                            name="Moderate",
                            stackgroup="one",
                            line=dict(width=1, color="rgba(255, 165, 0, 0.8)")
                        )
                    )
                    
                    consolidation_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["strength_strong"],
                            mode="lines",
                            name="Strong",
                            stackgroup="one",
                            line=dict(width=1, color="rgba(0, 128, 0, 0.8)")
                        )
                    )
                    
                    consolidation_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["strength_permanent"],
                            mode="lines",
                            name="Permanent",
                            stackgroup="one",
                            line=dict(width=1, color="rgba(0, 0, 255, 0.8)")
                        )
                    )
                    
                    consolidation_fig.update_layout(
                        title="Memory Consolidation Over Time",
                        xaxis_title="Time",
                        yaxis_title="Number of Memories",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                
                return memory_stats_display, memory_fig, consolidation_fig
                
            except Exception as e:
                logger.error(f"Error updating memory stats: {e}")
                return html.Div(f"Error: {e}"), go.Figure(), go.Figure()
        
        @self.app.callback(
            [Output("memory-network-graph", "figure"),
             Output("working-memory", "children"),
             Output("working-memory-graph", "figure"),
             Output("retrieval-stats-graph", "figure")],
            [Input("memory-network-interval", "n_intervals")]
        )
        def update_memory_network(n):
            if not self.lmm:
                return {}, html.Div("No data available"), {}, {}
            
            try:
                # Get memory module
                memory_module = None
                if hasattr(self.lmm, "memory_module"):
                    memory_module = self.lmm.memory_module
                
                # Get memory network graph data
                memory_graph_data = {}
                if memory_module:
                    # Get memory graph data through the memory module
                    memory_graph_result = memory_module.process({
                        "operation": "get_memory_graph",
                        "parameters": {"limit": 50}  # Limit to 50 nodes for visualization
                    })
                    
                    if memory_graph_result.get("success"):
                        memory_graph_data = memory_graph_result.get("graph", {})
                    
                    # Get working memory contents
                    working_memory_result = memory_module.process({
                        "operation": "get_working_memory"
                    })
                    
                    working_memory_contents = working_memory_result.get("contents", [])
                    
                    # Store working memory history
                    self.working_memory_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "capacity": working_memory_result.get("capacity", 0),
                        "usage": working_memory_result.get("usage", 0)
                    })
                    
                    # Limit history size
                    if len(self.working_memory_history) > 100:
                        self.working_memory_history = self.working_memory_history[-100:]
                
                # Create memory network graph
                network_fig = go.Figure()
                
                if memory_graph_data and "nodes" in memory_graph_data and "edges" in memory_graph_data:
                    nodes = memory_graph_data["nodes"]
                    edges = memory_graph_data["edges"]
                    
                    # Create a networkx graph
                    G = nx.Graph()
                    
                    # Add nodes with attributes
                    for node in nodes:
                        G.add_node(
                            node["id"],
                            content=node.get("content", ""),
                            type=node.get("type", ""),
                            activation=node.get("activation", 0.0),
                            strength=node.get("strength", 0.0)
                        )
                    
                    # Add edges with weights
                    for edge in edges:
                        G.add_edge(
                            edge["source"],
                            edge["target"],
                            weight=edge.get("weight", 0.5)
                        )
                    
                    # Use a layout algorithm to position nodes
                    pos = nx.spring_layout(G)
                    
                    # Create edge trace
                    edge_trace = go.Scatter(
                        x=[],
                        y=[],
                        line=dict(width=0.5, color="#888"),
                        hoverinfo="none",
                        mode="lines"
                    )
                    
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace["x"] += (x0, x1, None)
                        edge_trace["y"] += (y0, y1, None)
                    
                    # Create node trace
                    node_trace = go.Scatter(
                        x=[],
                        y=[],
                        text=[],
                        mode="markers",
                        hoverinfo="text",
                        marker=dict(
                            showscale=True,
                            colorscale="YlGnBu",
                            size=[],
                            color=[],
                            colorbar=dict(
                                thickness=15,
                                title="Memory Strength",
                                xanchor="left",
                                titleside="right"
                            ),
                            line=dict(width=2)
                        )
                    )
                    
                    # Add node positions and attributes
                    for node in G.nodes():
                        x, y = pos[node]
                        node_trace["x"] += (x,)
                        node_trace["y"] += (y,)
                        
                        # Get node attributes
                        content = G.nodes[node].get("content", "")
                        node_type = G.nodes[node].get("type", "")
                        activation = G.nodes[node].get("activation", 0.0)
                        strength = G.nodes[node].get("strength", 0.0)
                        
                        # Set node size based on activation
                        node_trace["marker"]["size"] += (10 + 20 * activation,)
                        
                        # Set node color based on strength
                        node_trace["marker"]["color"] += (strength,)
                        
                        # Set hover text
                        node_trace["text"] += (f"ID: {node}<br>Content: {content}<br>Type: {node_type}<br>Activation: {activation:.2f}<br>Strength: {strength:.2f}",)
                    
                    # Create the figure
                    network_fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Memory Association Network",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    )
                
                # Create working memory display
                working_memory_items = []
                
                if "working_memory_contents" in locals():
                    for i, memory in enumerate(working_memory_contents):
                        content = memory.get("content", "")
                        memory_type = memory.get("type", "")
                        activation = memory.get("activation", 0.0)
                        strength = memory.get("strength", 0.0)
                        
                        # Truncate long content
                        if len(content) > 50:
                            content = content[:47] + "..."
                        
                        working_memory_items.append(
                            html.Div([
                                html.Strong(f"{i+1}. "),
                                html.Span(content),
                                html.Br(),
                                html.Small(f"Type: {memory_type}, Activation: {activation:.2f}, Strength: {strength:.2f}")
                            ], style={"margin": "5px 0", "padding": "5px", "border": "1px solid #ddd", "borderRadius": "5px"})
                        )
                
                if not working_memory_items:
                    working_memory_items.append(html.Div("No items in working memory"))
                
                working_memory_display = html.Div(working_memory_items)
                
                # Create working memory graph
                working_memory_fig = go.Figure()
                
                if self.working_memory_history:
                    df = pd.DataFrame(self.working_memory_history)
                    
                    # Format timestamp
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    working_memory_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["usage"],
                            mode="lines+markers",
                            name="Usage"
                        )
                    )
                    
                    working_memory_fig.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["capacity"],
                            mode="lines",
                            name="Capacity",
                            line=dict(dash="dash")
                        )
                    )
                    
                    working_memory_fig.update_layout(
                        title="Working Memory Usage Over Time",
                        xaxis_title="Time",
                        yaxis_title="Items",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                
                # Create retrieval stats graph
                retrieval_fig = go.Figure()
                
                if self.retrieval_stats["timestamps"]:
                    retrieval_fig.add_trace(
                        go.Scatter(
                            x=self.retrieval_stats["timestamps"],
                            y=self.retrieval_stats["counts"],
                            mode="lines+markers",
                            name="Retrieval Count"
                        )
                    )
                    
                    retrieval_fig.add_trace(
                        go.Scatter(
                            x=self.retrieval_stats["timestamps"],
                            y=self.retrieval_stats["scores"],
                            mode="lines+markers",
                            name="Avg Score",
                            yaxis="y2"
                        )
                    )
                    
                    retrieval_fig.update_layout(
                        title="Memory Retrieval Statistics",
                        xaxis_title="Time",
                        yaxis_title="Count",
                        yaxis2=dict(
                            title="Avg Score",
                            overlaying="y",
                            side="right",
                            range=[0, 1]
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                
                return network_fig, working_memory_display, working_memory_fig, retrieval_fig
                
            except Exception as e:
                logger.error(f"Error updating memory network: {e}")
                return go.Figure(), html.Div(f"Error: {e}"), go.Figure(), go.Figure()
        
        @self.app.callback(
            Output("conversation-history", "children"),
            [Input("conversation-interval", "n_intervals")]
        )
        def update_conversation_history(n):
            if not self.lmm:
                return html.Div("No conversation history available")
            
            try:
                # Get memory module
                memory_module = None
                if hasattr(self.lmm, "memory_module"):
                    memory_module = self.lmm.memory_module
                
                conversation_memories = []
                if memory_module:
                    # Search for conversation memories
                    search_result = memory_module.process({
                        "operation": "search",
                        "parameters": {
                            "query": "conversation",
                            "context_tags": ["conversation", "user_message", "lmm_response"],
                            "limit": 10,
                            "retrieval_strategy": "context"
                        }
                    })
                    
                    if search_result.get("success"):
                        conversation_memories = search_result.get("memories", [])
                
                # Create conversation history display
                conversation_items = []
                
                for memory in sorted(conversation_memories, key=lambda x: x.get("created_at", ""), reverse=True):
                    content = memory.get("content", "")
                    created_at = memory.get("created_at", "")
                    
                    # Try to parse the timestamp
                    try:
                        dt = datetime.fromisoformat(created_at)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp = created_at
                    
                    conversation_items.append(
                        html.Div([
                            html.Small(timestamp, style={"color": "#666"}),
                            html.Br(),
                            html.Div(content)
                        ], style={"margin": "10px 0", "padding": "10px", "borderBottom": "1px solid #eee"})
                    )
                
                if not conversation_items:
                    conversation_items.append(html.Div("No conversation history"))
                
                return html.Div(conversation_items)
                
            except Exception as e:
                logger.error(f"Error updating conversation history: {e}")
                return html.Div(f"Error: {e}")
    
    def start(self, debug: bool = False):
        """
        Start the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting dashboard server on port {self.port}")
        self.app.run_server(debug=debug, port=self.port)
    
    def start_background(self):
        """Start the dashboard server in a background thread."""
        if self.thread and self.thread.is_alive():
            logger.warning("Dashboard already running in background")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.start)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started dashboard in background thread on port {self.port}")
    
    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        logger.info("Stopped dashboard server")

def launch_dashboard(lmm_instance=None, port: int = 8050):
    """
    Launch the development dashboard.
    
    Args:
        lmm_instance: Instance of the LMM
        port: Port to run the dashboard on
    """
    dashboard = DevelopmentDashboard(lmm_instance, port)
    dashboard.start_background()
    return dashboard

if __name__ == "__main__":
    # Launch dashboard without LMM instance (for testing)
    launch_dashboard() 