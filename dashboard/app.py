"""
Dashboard application for the Neural Child project.

This module contains a Dash application that provides a visual interface
for monitoring and interacting with the Neural Child.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Add parent directory to path to import neural_child modules
sys.path.append(str(Path(__file__).parent.parent))

from neural_child.mind.mind import Mind

# Initialize the Mind
mind = Mind(
    initial_age_months=0.0,
    development_speed=10.0,  # Faster speed for demo purposes
)

# Initialize app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Neural Child Dashboard"
)

# Define colors
COLORS = {
    "background": "#222222",
    "card_background": "#333333",
    "text": "#FFFFFF",
    "primary": "#375A7F",
    "success": "#00bc8c",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#3498DB",
    "accent": "#9B59B6",
    # Domain-specific colors
    "language": "#3498DB",    # Blue
    "emotion": "#E74C3C",     # Red
    "cognition": "#9B59B6",   # Purple
    "social": "#00bc8c",      # Green
    "memory": "#F39C12",      # Orange
    "development": "#375A7F", # Dark Blue
}

# Define the app layout
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": COLORS["background"], "color": COLORS["text"], "minHeight": "100vh"},
    children=[
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
                                dbc.Col(dbc.NavbarBrand("Neural Child", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                                dbc.NavItem(dbc.NavLink("Development", href="#")),
                                dbc.NavItem(dbc.NavLink("Interaction", href="#")),
                                dbc.NavItem(dbc.NavLink("Settings", href="#")),
                            ],
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]
            ),
            color=COLORS["primary"],
            dark=True,
            className="mb-4",
        ),
        
        # Main content
        dbc.Row([
            # Left column - Child status
            dbc.Col([
                # Child overview card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Neural Child Status", className="card-title"),
                        html.Div(id="child-age-display", className="mb-2"),
                        html.Div(id="developmental-stage-display", className="mb-3"),
                        
                        # Age controls
                        dbc.Row([
                            dbc.Col([
                                html.Label("Development Speed:"),
                                dcc.Slider(
                                    id="development-speed-slider",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=10,
                                    marks={0: '0x', 25: '25x', 50: '50x', 75: '75x', 100: '100x'},
                                ),
                            ]),
                            dbc.Col([
                                html.Br(),
                                dbc.Button(
                                    "Apply Speed", 
                                    id="apply-speed-btn", 
                                    color="primary", 
                                    className="mt-2"
                                ),
                            ]),
                        ]),
                        
                        # Advance time button
                        dbc.Row([
                            dbc.Col([
                                html.Label("Advance Time:"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="advance-time-input",
                                        type="number",
                                        placeholder="Hours",
                                        value=1,
                                        min=0,
                                        step=1,
                                    ),
                                    dbc.InputGroupText("hours"),
                                    dbc.Button(
                                        "Advance", 
                                        id="advance-time-btn", 
                                        color="primary"
                                    ),
                                ]),
                            ]),
                        ], className="mt-3"),
                        
                        # Save/Load controls
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Save State", 
                                    id="save-state-btn", 
                                    color="success", 
                                    className="mt-3 me-2"
                                ),
                                dbc.Button(
                                    "Load State", 
                                    id="load-state-btn", 
                                    color="info", 
                                    className="mt-3"
                                ),
                                html.Div(id="save-load-status", className="mt-2"),
                            ]),
                        ]),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
                
                # Needs card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Physiological Needs", className="card-title"),
                        html.Div(id="needs-display"),
                        
                        # Physiological events
                        html.H5("Simulate Events", className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Feeding", 
                                    id="feeding-btn", 
                                    color="primary", 
                                    className="me-2 mb-2"
                                ),
                                dbc.Button(
                                    "Diaper Change", 
                                    id="diaper-btn", 
                                    color="primary", 
                                    className="me-2 mb-2"
                                ),
                                dbc.Button(
                                    "Sleep", 
                                    id="sleep-btn", 
                                    color="primary", 
                                    className="me-2 mb-2"
                                ),
                                dbc.Button(
                                    "Play", 
                                    id="play-btn", 
                                    color="primary", 
                                    className="me-2 mb-2"
                                ),
                            ]),
                        ]),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
                
                # Emotional state card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Emotional State", className="card-title"),
                        dcc.Graph(id="emotion-chart"),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
            ], md=4),
            
            # Middle column - Development
            dbc.Col([
                # Development metrics card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Development Progress", className="card-title"),
                        
                        # Tabs for different development domains
                        dbc.Tabs([
                            dbc.Tab(
                                dcc.Graph(id="language-dev-chart"),
                                label="Language", 
                                tab_id="language-tab",
                                label_style={"color": COLORS["language"]},
                            ),
                            dbc.Tab(
                                dcc.Graph(id="emotional-dev-chart"),
                                label="Emotional", 
                                tab_id="emotional-tab",
                                label_style={"color": COLORS["emotion"]},
                            ),
                            dbc.Tab(
                                dcc.Graph(id="cognitive-dev-chart"),
                                label="Cognitive", 
                                tab_id="cognitive-tab",
                                label_style={"color": COLORS["cognition"]},
                            ),
                            dbc.Tab(
                                dcc.Graph(id="social-dev-chart"),
                                label="Social", 
                                tab_id="social-tab",
                                label_style={"color": COLORS["social"]},
                            ),
                            dbc.Tab(
                                dcc.Graph(id="memory-dev-chart"),
                                label="Memory", 
                                tab_id="memory-tab",
                                label_style={"color": COLORS["memory"]},
                            ),
                        ], id="dev-tabs"),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
                
                # Milestones card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Developmental Milestones", className="card-title"),
                        html.Div(id="milestones-display"),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
                
                # Memory Statistics
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Memory Statistics", className="card-title"),
                        dcc.Graph(id="memory-stats-chart"),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
            ], md=4),
            
            # Right column - Interaction
            dbc.Col([
                # Interaction card
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Interact with Child", className="card-title"),
                        
                        # Mother's input
                        html.Label("Mother's utterance:"),
                        dbc.Textarea(
                            id="mother-utterance-input",
                            placeholder="Enter what to say to the child...",
                            style={"height": "100px"},
                            className="mb-2",
                        ),
                        
                        # Mother's emotional state
                        html.Label("Mother's emotional state:"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Joy:"),
                                dcc.Slider(
                                    id="joy-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0.5,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Sadness:"),
                                dcc.Slider(
                                    id="sadness-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                        ], className="mb-2"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Anger:"),
                                dcc.Slider(
                                    id="anger-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Fear:"),
                                dcc.Slider(
                                    id="fear-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                        ], className="mb-2"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Surprise:"),
                                dcc.Slider(
                                    id="surprise-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Interest:"),
                                dcc.Slider(
                                    id="interest-slider",
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    value=0.5,
                                    marks={0: '0', 0.5: '0.5', 1: '1'},
                                ),
                            ], width=6),
                        ], className="mb-3"),
                        
                        # Teaching elements
                        html.Label("Teaching elements (optional):"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Topic:"),
                            dbc.Input(id="teaching-topic-input", placeholder="e.g., colors, animals"),
                        ], className="mb-2"),
                        
                        dbc.InputGroup([
                            dbc.InputGroupText("Concept:"),
                            dbc.Input(id="teaching-concept-input", placeholder="e.g., red, dog"),
                        ], className="mb-3"),
                        
                        # Context
                        html.Label("Context (optional):"),
                        dbc.Textarea(
                            id="context-input",
                            placeholder="Enter context (e.g., location, objects present)...",
                            style={"height": "60px"},
                            className="mb-3",
                        ),
                        
                        # Send button
                        dbc.Button(
                            "Send to Child", 
                            id="send-interaction-btn", 
                            color="primary", 
                            size="lg",
                            className="w-100 mb-3",
                        ),
                        
                        # Conversation display
                        html.H5("Conversation"),
                        html.Div(id="conversation-display", style={"maxHeight": "400px", "overflowY": "auto"}),
                    ]),
                    className="mb-4",
                    style={"backgroundColor": COLORS["card_background"]},
                ),
            ], md=4),
        ]),
        
        # Update interval for passive changes (needs decay, etc.)
        dcc.Interval(
            id="update-interval",
            interval=5000,  # in milliseconds
            n_intervals=0
        ),
        
        # Store components for maintaining state
        dcc.Store(id="conversation-store", data=[]),
        dcc.Store(id="development-history-store", data=[]),
    ]
)

# Define callback for updating child age and developmental stage display
@callback(
    [Output("child-age-display", "children"),
     Output("developmental-stage-display", "children")],
    [Input("update-interval", "n_intervals"),
     Input("advance-time-btn", "n_clicks"),
     Input("apply-speed-btn", "n_clicks"),
     Input("send-interaction-btn", "n_clicks"),
     Input("feeding-btn", "n_clicks"),
     Input("diaper-btn", "n_clicks"),
     Input("sleep-btn", "n_clicks"),
     Input("play-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_child_status(n_intervals, advance_click, speed_click, interact_click, 
                        feeding_click, diaper_click, sleep_click, play_click):
    # Get the current state
    snapshot = mind.get_state_snapshot()
    
    # Format age
    age_months = snapshot["mind_state"]["age_months"]
    years = int(age_months // 12)
    months = int(age_months % 12)
    days = int((age_months % 1) * 30)
    
    if years > 0:
        age_str = f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}, {days} day{'s' if days != 1 else ''}"
    else:
        age_str = f"{months} month{'s' if months != 1 else ''}, {days} day{'s' if days != 1 else ''}"
    
    age_display = [
        html.Span("Age: ", className="fw-bold"),
        html.Span(age_str)
    ]
    
    # Format developmental stage
    stage = snapshot["mind_state"]["developmental_stage"]
    dev_stage_display = [
        html.Span("Developmental Stage: ", className="fw-bold"),
        html.Span(stage)
    ]
    
    return age_display, dev_stage_display

# Define callback for updating needs display
@callback(
    Output("needs-display", "children"),
    [Input("update-interval", "n_intervals"),
     Input("advance-time-btn", "n_clicks"),
     Input("feeding-btn", "n_clicks"),
     Input("diaper-btn", "n_clicks"),
     Input("sleep-btn", "n_clicks"),
     Input("play-btn", "n_clicks"),
     Input("send-interaction-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_needs_display(n_intervals, advance_click, feeding_click, 
                         diaper_click, sleep_click, play_click, interact_click):
    # Get the current needs
    snapshot = mind.get_state_snapshot()
    needs = snapshot["needs"]
    
    # Create progress bars for each need
    needs_display = []
    
    for need, value in needs.items():
        # Determine progress bar color based on value
        if need == "stimulation":
            # For stimulation, optimal is in the middle
            if 0.4 <= value <= 0.6:
                color = "success"
            elif (0.2 <= value < 0.4) or (0.6 < value <= 0.8):
                color = "warning"
            else:
                color = "danger"
        else:
            # For other needs, lower is better
            if value <= 0.3:
                color = "success"
            elif value <= 0.7:
                color = "warning"
            else:
                color = "danger"
        
        # Format label
        need_label = need.replace("_", " ").title()
        
        # Create progress bar
        progress_bar = dbc.Progress(
            value=value * 100,
            color=color,
            className="mb-2",
            label=f"{need_label}: {value:.2f}",
            style={"height": "30px"},
        )
        
        needs_display.append(progress_bar)
    
    return needs_display

# Define callback for updating emotion chart
@callback(
    Output("emotion-chart", "figure"),
    [Input("update-interval", "n_intervals"),
     Input("send-interaction-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_emotion_chart(n_intervals, interact_click):
    # Get the current emotional state
    snapshot = mind.get_state_snapshot()
    emotional_state = snapshot["emotional_state"]
    
    # Create data for polar chart
    emotions = list(emotional_state.keys())
    values = list(emotional_state.values())
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Barpolar(
        r=values,
        theta=emotions,
        marker_color=px.colors.sequential.Plasma_r[:len(emotions)],
        marker_line_color="white",
        marker_line_width=1,
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    return fig

# Define callback for updating development charts
@callback(
    [Output("language-dev-chart", "figure"),
     Output("emotional-dev-chart", "figure"),
     Output("cognitive-dev-chart", "figure"),
     Output("social-dev-chart", "figure"),
     Output("memory-dev-chart", "figure")],
    [Input("update-interval", "n_intervals"),
     Input("dev-tabs", "active_tab"),
     Input("send-interaction-btn", "n_clicks"),
     Input("advance-time-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_development_charts(n_intervals, active_tab, interact_click, advance_click):
    # Get the development summary
    summary = mind.get_developmental_summary()
    metrics = summary["metrics"]
    
    # Create the charts
    charts = {}
    
    # Language development chart
    lang_metrics = metrics["language"]
    lang_fig = go.Figure()
    
    lang_fig.add_trace(go.Bar(
        x=list(lang_metrics.keys()),
        y=list(lang_metrics.values()),
        marker_color=COLORS["language"],
        opacity=0.8
    ))
    
    lang_fig.update_layout(
        title="Language Development",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    # Emotional development chart
    emo_metrics = metrics["emotional"]
    emo_fig = go.Figure()
    
    emo_fig.add_trace(go.Bar(
        x=list(emo_metrics.keys()),
        y=list(emo_metrics.values()),
        marker_color=COLORS["emotion"],
        opacity=0.8
    ))
    
    emo_fig.update_layout(
        title="Emotional Development",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    # Cognitive development chart
    cog_metrics = metrics["cognitive"]
    cog_fig = go.Figure()
    
    cog_fig.add_trace(go.Bar(
        x=list(cog_metrics.keys()),
        y=list(cog_metrics.values()),
        marker_color=COLORS["cognition"],
        opacity=0.8
    ))
    
    cog_fig.update_layout(
        title="Cognitive Development",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    # Social development chart
    soc_metrics = metrics["social"]
    soc_fig = go.Figure()
    
    soc_fig.add_trace(go.Bar(
        x=list(soc_metrics.keys()),
        y=list(soc_metrics.values()),
        marker_color=COLORS["social"],
        opacity=0.8
    ))
    
    soc_fig.update_layout(
        title="Social Development",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    # Memory development chart
    mem_metrics = metrics["memory"]
    mem_fig = go.Figure()
    
    mem_fig.add_trace(go.Bar(
        x=list(mem_metrics.keys()),
        y=list(mem_metrics.values()),
        marker_color=COLORS["memory"],
        opacity=0.8
    ))
    
    mem_fig.update_layout(
        title="Memory Development",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    return lang_fig, emo_fig, cog_fig, soc_fig, mem_fig

# Define callback for updating milestones display
@callback(
    Output("milestones-display", "children"),
    [Input("update-interval", "n_intervals"),
     Input("advance-time-btn", "n_clicks"),
     Input("send-interaction-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_milestones_display(n_intervals, advance_click, interact_click):
    # Get the development summary
    summary = mind.get_developmental_summary()
    milestones = summary["milestones"]
    
    # Create milestones display
    milestones_display = []
    
    for category, achieved in milestones.items():
        if achieved:
            # Add category header
            milestones_display.append(
                html.H5(category.title(), className="mt-2", 
                        style={"color": COLORS.get(category.lower(), COLORS["text"])})
            )
            
            # Add milestones as a list
            milestone_items = []
            for milestone in achieved:
                milestone_items.append(html.Li(milestone))
            
            milestones_display.append(html.Ul(milestone_items))
    
    if not milestones_display:
        milestones_display = [html.P("No milestones achieved yet.")]
    
    return milestones_display

# Define callback for updating memory statistics chart
@callback(
    Output("memory-stats-chart", "figure"),
    [Input("update-interval", "n_intervals"),
     Input("send-interaction-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_memory_stats_chart(n_intervals, interact_click):
    # Get the memory counts
    snapshot = mind.get_state_snapshot()
    memory_counts = snapshot["memory_counts"]
    
    # Create data for pie chart
    labels = list(memory_counts.keys())
    values = list(memory_counts.values())
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=[COLORS.get("memory"), COLORS.get("language"), COLORS.get("cognition")],
            line=dict(color=COLORS["text"], width=1)
        ),
        textinfo="label+percent",
        insidetextorientation="radial",
        hole=0.4
    ))
    
    fig.update_layout(
        title="Memory Types",
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=COLORS["card_background"],
        plot_bgcolor=COLORS["card_background"],
        font_color=COLORS["text"],
    )
    
    return fig

# Define callback for conversation
@callback(
    [Output("conversation-display", "children"),
     Output("conversation-store", "data")],
    [Input("send-interaction-btn", "n_clicks")],
    [State("mother-utterance-input", "value"),
     State("joy-slider", "value"),
     State("sadness-slider", "value"),
     State("anger-slider", "value"),
     State("fear-slider", "value"),
     State("surprise-slider", "value"),
     State("interest-slider", "value"),
     State("teaching-topic-input", "value"),
     State("teaching-concept-input", "value"),
     State("context-input", "value"),
     State("conversation-store", "data")],
    prevent_initial_call=True
)
def update_conversation(n_clicks, utterance, joy, sadness, anger, fear, surprise, interest, 
                       topic, concept, context, conversation_data):
    if not utterance:
        return dash.no_update, dash.no_update
    
    # Create emotional state
    emotional_state = {}
    if joy > 0:
        emotional_state["joy"] = joy
    if sadness > 0:
        emotional_state["sadness"] = sadness
    if anger > 0:
        emotional_state["anger"] = anger
    if fear > 0:
        emotional_state["fear"] = fear
    if surprise > 0:
        emotional_state["surprise"] = surprise
    if interest > 0:
        emotional_state["interest"] = interest
    
    # Create teaching elements
    teaching_elements = {}
    if topic and concept:
        teaching_elements = {
            "topic": topic,
            "concept": concept
        }
    
    # Create context
    context_dict = {}
    if context:
        context_dict["description"] = context
    
    # Process interaction
    interaction_data = {
        "utterance": utterance,
        "emotional_state": emotional_state,
        "teaching_elements": teaching_elements,
        "context": context_dict
    }
    
    # Get response from mind
    response = mind.process_mother_interaction(interaction_data)
    
    # Add to conversation history
    timestamp = datetime.now().strftime("%H:%M:%S")
    new_entry = {
        "timestamp": timestamp,
        "mother": utterance,
        "mother_emotional_state": emotional_state,
        "child": response["utterance"],
        "child_emotional_state": response["emotional_state"],
        "understanding_level": response.get("understanding_level", 0.0)
    }
    
    conversation_data.append(new_entry)
    
    # Create conversation display
    conversation_elements = []
    
    for entry in conversation_data:
        # Add mother's message
        mother_emotions = ", ".join([f"{k}: {v:.1f}" for k, v in entry["mother_emotional_state"].items() if v > 0])
        mother_emotion_badge = html.Span(mother_emotions, className="badge bg-primary ms-2") if mother_emotions else None
        
        conversation_elements.append(
            html.Div([
                html.Span(f"[{entry['timestamp']}] ", className="text-muted"),
                html.Span("Mother: ", className="fw-bold text-primary"),
                html.Span(entry["mother"]),
                mother_emotion_badge
            ], className="mb-2")
        )
        
        # Add child's message
        child_emotions = ", ".join([f"{k}: {v:.1f}" for k, v in entry["child_emotional_state"].items() if v > 0])
        child_emotion_badge = html.Span(child_emotions, className="badge bg-success ms-2") if child_emotions else None
        
        # Determine understanding badge color
        understanding = entry.get("understanding_level", 0.0)
        if understanding < 0.3:
            understanding_color = "danger"
        elif understanding < 0.7:
            understanding_color = "warning"
        else:
            understanding_color = "success"
        
        understanding_badge = html.Span(
            f"Understanding: {understanding:.1f}", 
            className=f"badge bg-{understanding_color} ms-2"
        )
        
        conversation_elements.append(
            html.Div([
                html.Span(f"[{entry['timestamp']}] ", className="text-muted"),
                html.Span("Child: ", className="fw-bold text-success"),
                html.Span(entry["child"]),
                child_emotion_badge,
                understanding_badge
            ], className="mb-3")
        )
    
    return conversation_elements, conversation_data

# Define callback for handling physiological events
@callback(
    Output("needs-display", "children", allow_duplicate=True),
    [Input("feeding-btn", "n_clicks"),
     Input("diaper-btn", "n_clicks"),
     Input("sleep-btn", "n_clicks"),
     Input("play-btn", "n_clicks")],
    prevent_initial_call=True
)
def handle_physiological_events(feeding_click, diaper_click, sleep_click, play_click):
    # Determine which button was clicked
    button_id = ctx.triggered_id
    
    if button_id == "feeding-btn":
        mind.simulate_physiological_event("feeding", 0.8)
    elif button_id == "diaper-btn":
        mind.simulate_physiological_event("diaper_change", 0.7)
    elif button_id == "sleep-btn":
        mind.simulate_physiological_event("sleep", 0.9)
    elif button_id == "play-btn":
        mind.simulate_physiological_event("play", 0.7)
    
    # Return a placeholder - the actual update will be handled by update_needs_display
    return dash.no_update

# Define callback for advancing time
@callback(
    Output("child-age-display", "children", allow_duplicate=True),
    [Input("advance-time-btn", "n_clicks")],
    [State("advance-time-input", "value")],
    prevent_initial_call=True
)
def advance_time(n_clicks, hours):
    if hours and hours > 0:
        # Convert hours to seconds
        seconds = hours * 3600
        
        # Update mind
        mind.update(seconds)
    
    # Return a placeholder - the actual update will be handled by update_child_status
    return dash.no_update

# Define callback for applying development speed
@callback(
    Output("child-age-display", "children", allow_duplicate=True),
    [Input("apply-speed-btn", "n_clicks")],
    [State("development-speed-slider", "value")],
    prevent_initial_call=True
)
def apply_development_speed(n_clicks, speed):
    if speed is not None:
        # Update development speed
        mind.development.development_speed = speed
    
    # Return a placeholder - the actual update will be handled by update_child_status
    return dash.no_update

# Define callback for saving state
@callback(
    Output("save-load-status", "children"),
    [Input("save-state-btn", "n_clicks"),
     Input("load-state-btn", "n_clicks")],
    [State("save-load-status", "children")],
    prevent_initial_call=True
)
def handle_save_load(save_clicks, load_clicks, current_status):
    button_id = ctx.triggered_id
    
    if button_id == "save-state-btn":
        # Save the state
        save_dir = mind.save()
        if save_dir:
            return html.Div([
                html.Span("✅ State saved to: "),
                html.Span(save_dir, style={"fontWeight": "bold"})
            ], className="text-success")
        else:
            return html.Div("❌ Error saving state", className="text-danger")
            
    elif button_id == "load-state-btn":
        # Currently just a placeholder - would need a file picker in a real implementation
        return html.Div("Load functionality requires a file picker dialog", className="text-warning")
    
    return current_status

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050) 