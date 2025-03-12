"""Reusable components for the AgenDev Dashboard."""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .styles import styles

def create_chat_message(sender, message, timestamp=None, message_type="text"):
    """Create a styled message for the chat interface."""
    is_system = sender == "AgenDev"
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    message_style = {
        "padding": "12px 18px",
        "borderRadius": "18px",
        "margin": "8px 0",
        "maxWidth": "85%",
        "wordWrap": "break-word",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)",
    }
    
    if is_system:
        # System message (AgenDev) - blue gradient background
        message_style.update({
            "background": "linear-gradient(135deg, #4c7fff 0%, #6a5acd 100%)",
            "color": "white",
            "marginLeft": "auto",
            "marginRight": "15px",
            "borderTopRightRadius": "4px",
        })
        sender_class = "text-info"
    else:
        # User message - purple gradient background
        message_style.update({
            "background": "linear-gradient(135deg, #9f7aea 0%, #6a5acd 100%)",
            "color": "white",
            "marginRight": "auto",
            "marginLeft": "15px",
            "borderTopLeftRadius": "4px",
        })
        sender_class = "text-light"
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span(sender, className=f"fw-bold {sender_class} me-2"),
                html.Small(timestamp, className="text-light opacity-75")
            ], className="d-flex justify-content-between align-items-center mb-1"),
            html.Div(message)
        ], style=message_style)
    ], className="d-flex")

def create_chat_interface():
    """Create the chat interface."""
    return html.Div([
        # Chat header
        html.Div([
            html.H4("AgenDev Assistant", className="mb-0"),
            html.Div([
                html.Span("Status: ", className="text-muted me-1"),
                dbc.Badge("Online", color="success", className="me-2")
            ])
        ], className="d-flex justify-content-between align-items-center p-3 border-bottom border-secondary"),
        
        # Chat messages container
        html.Div(id="chat-messages", className="px-2 py-3", style={
            "height": "calc(100vh - 180px)",
            "overflowY": "auto"
        }),
        
        # Chat input area
        html.Div([
            dbc.InputGroup([
                dbc.Textarea(
                    id="chat-input",
                    placeholder="Type a message...",
                    className="border-0",
                    style={"resize": "none", "height": "48px"}
                ),
                dbc.Button(
                    html.I(className="fas fa-paper-plane"),
                    id="send-message",
                    color="primary",
                    className="ms-2"
                )
            ])
        ], className="p-3 border-top border-secondary")
    ], className="h-100 d-flex flex-column", style={
        "background": styles['sidebar'],
        "borderRadius": "8px",
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
    })

def create_card(title, content, id=None, color="primary"):
    """Create a custom styled card component."""
    card_props = {
        "children": [
            dbc.CardHeader(
                html.H5(title, className="mb-0"),
                className="bg-transparent border-bottom border-dark"
            ),
            dbc.CardBody(content, className="p-3")
        ],
        "className": "mb-3 h-100",
        "style": {
            "background": styles['card'],
            "border": f"1px solid rgba({styles[color]}, 0.3)",
            "borderRadius": "8px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
        }
    }
    
    # Add id if provided
    if id is not None:
        card_props["id"] = id
    
    return dbc.Card(**card_props)

def create_tasks_section(agendev):
    """Create the tasks section with a table of tasks."""
    # Extract task data for the table
    tasks_data = []
    for task_id, task in agendev.task_graph.tasks.items():
        tasks_data.append({
            "id": str(task_id),
            "title": task.title,
            "status": task.status.value,
            "type": task.task_type.value,
            "priority": task.priority.value,
            "risk": task.risk.value,
            "duration": f"{task.estimated_duration_hours:.1f}h",
        })
    
    if tasks_data:
        tasks_table = dbc.Table.from_dataframe(
            pd.DataFrame(tasks_data)[["title", "status", "type", "priority", "duration"]],
            striped=True,
            bordered=False,
            hover=True,
            className="table-dark",
            style={"fontSize": "0.85rem"}
        )
    else:
        tasks_table = html.Div(
            html.P("No tasks found. Create a project to generate tasks."),
            className="text-center p-3"
        )
    
    return create_card("Tasks", tasks_table, color="accent_blue")

def create_implementation_section():
    """Create the implementation section showing code."""
    # File selector
    file_selector = html.Div([
        dbc.Label("Select Implementation"),
        dcc.Dropdown(
            id="implementation-file",
            options=[],
            value=None,
            className="mb-3 bg-dark text-white",
        )
    ])
    
    # Code viewer content
    code_viewer = html.Div(id="code-viewer", className="mt-3")
    
    # Combine selector and viewer
    content = html.Div([
        file_selector,
        code_viewer
    ])
    
    return create_card("Implementation", content, color="accent_purple")

def create_overview_section(agendev):
    """Create the overview section with status and metrics."""
    project_status = agendev.get_project_status()
    progress = project_status.get('progress', {}).get('percentage', 0)
    
    # Project metrics
    metrics = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2(f"{progress:.1f}%", className="mb-0 text-info"),
                    html.Small("Progress", className="text-muted")
                ], className="text-center p-2 border border-secondary rounded")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H2(project_status.get('tasks', {}).get('total', 0), className="mb-0 text-warning"),
                    html.Small("Total Tasks", className="text-muted")
                ], className="text-center p-2 border border-secondary rounded")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H2(project_status.get('tasks', {}).get('by_status', {}).get('completed', 0), className="mb-0 text-success"),
                    html.Small("Completed", className="text-muted")
                ], className="text-center p-2 border border-secondary rounded")
            ], width=4)
        ], className="mb-3"),
        
        # Status progress bar
        html.Div([
            html.P("Overall Progress", className="mb-1"),
            dbc.Progress(value=progress, color="info", style={"height": "10px"}),
        ], className="mb-3"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-play me-2"), "Start Process"],
                    id="btn-start-auto",
                    color="success",
                    className="w-100"
                )
            ], width=6),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-stop me-2"), "Stop Process"],
                    id="btn-stop-auto",
                    color="danger",
                    className="w-100"
                )
            ], width=6)
        ]),
        html.Div(id="auto-process-output", className="mt-2"),
    ])
    
    return create_card("Project Overview", metrics, color="info")

def create_project_form():
    """Create the project creation form."""
    return html.Div([
        dbc.Label("Project Name"),
        dbc.Input(id="project-name", type="text", placeholder="Enter project name", className="mb-2"),
        dbc.Label("Project Description"),
        dbc.Textarea(
            id="project-description",
            placeholder="Describe your project in detail...",
            style={"height": "120px"}
        ),
        dbc.Button(
            [html.I(className="fas fa-rocket me-2"), "Create Project"],
            id="btn-create-project",
            color="primary",
            className="mt-3 w-100"
        ),
        html.Div(id="project-creation-output", className="mt-2")
    ])

def create_advanced_features_section(agendev):
    """Create the advanced features section with previously unused modules."""
    return create_card(
        "Advanced Features",
        [
            # Tab navigation
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            html.Div([
                                html.P("Optimize the implementation plan based on different priorities."),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Optimization Goal"),
                                        dbc.Select(
                                            id="optimization-goal",
                                            options=[
                                                {"label": "Speed", "value": "speed"},
                                                {"label": "Quality", "value": "quality"},
                                                {"label": "Risk", "value": "risk"},
                                            ],
                                            value="speed",
                                            className="mb-3"
                                        ),
                                        dbc.Button(
                                            "Generate Alternative Plan", 
                                            id="generate-alternative-plan-button", 
                                            color="primary",
                                            className="mb-3"
                                        ),
                                    ]),
                                ]),
                                # Results section
                                html.Div(id="alternative-plan-results", className="mt-3"),
                            ]),
                        ],
                        label="Alternative Planning",
                        tab_id="tab-alternative-planning",
                    ),
                    dbc.Tab(
                        [
                            html.Div([
                                html.P("Code context analysis for improved understanding."),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Build Code Context", 
                                            id="build-context-button", 
                                            color="primary",
                                            className="mb-3"
                                        ),
                                    ]),
                                ]),
                                # Results section
                                html.Div(id="context-results", className="mt-3"),
                            ]),
                        ],
                        label="Code Context",
                        tab_id="tab-code-context",
                    ),
                    dbc.Tab(
                        [
                            html.Div([
                                html.P("Analyze project completion probability and risk factors."),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Analyze Risk", 
                                            id="analyze-risk-button", 
                                            color="primary",
                                            className="mb-3"
                                        ),
                                    ]),
                                ]),
                                # Results section
                                html.Div(id="risk-analysis-results", className="mt-3"),
                            ]),
                        ],
                        label="Risk Analysis",
                        tab_id="tab-risk-analysis",
                    ),
                    dbc.Tab(
                        [
                            html.Div([
                                html.P("Manage and view code snapshots."),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Refresh Snapshots", 
                                            id="refresh-snapshots-button", 
                                            color="primary",
                                            className="mb-3"
                                        ),
                                    ]),
                                ]),
                                # List of snapshots
                                html.Div(id="snapshots-list", className="mt-3"),
                            ]),
                        ],
                        label="Code Snapshots",
                        tab_id="tab-snapshots",
                    ),
                    dbc.Tab(
                        [
                            html.Div([
                                html.P("Manage and run automatic tests for your code."),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Generate Tests", 
                                            id="generate-tests-button", 
                                            color="primary",
                                            className="mb-3"
                                        ),
                                        dbc.Input(
                                            id="test-path-input",
                                            type="text",
                                            placeholder="Path to file for testing",
                                            className="mb-3"
                                        ),
                                    ]),
                                ]),
                                # Test results section
                                html.Div(id="test-results", className="mt-3"),
                            ]),
                        ],
                        label="Test Generation",
                        tab_id="tab-test-generation",
                    ),
                ],
                id="advanced-features-tabs",
                active_tab="tab-alternative-planning",
            ),
        ],
        id="advanced-features-card",
        color="accent_blue"
    ) 