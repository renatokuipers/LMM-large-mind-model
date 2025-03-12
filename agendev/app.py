import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from uuid import UUID
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import time
from datetime import datetime

# Import AgenDev components
from agendev.core import AgenDev, AgenDevConfig, ProjectState
from agendev.models.task_models import TaskType, TaskStatus, TaskPriority, TaskRisk
from agendev.tts_notification import NotificationPriority, NotificationType
from agendev.utils.fs_utils import resolve_path
from agendev.llm_module import LLMClient
from agendev.tts_module import TTSClient

# Initialize the AgenDev system
try:
    config = AgenDevConfig(
        project_name="AgenDev Dashboard",
        llm_base_url="http://192.168.2.12:1234",
        tts_base_url="http://127.0.0.1:7860",
        notifications_enabled=True
    )
    agendev = AgenDev(config)
except Exception as e:
    print(f"Warning: Error initializing AgenDev: {e}")
    # Create a minimal instance for UI rendering
    config = AgenDevConfig(project_name="AgenDev Dashboard")
    agendev = AgenDev(config)

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
app.title = "AgenDev Dashboard"
server = app.server

# Custom styles for dark theme with blue/purple accents
styles = {
    'background': 'linear-gradient(135deg, #1e2030 0%, #2d3250 100%)',
    'card_bg': 'linear-gradient(135deg, rgba(30, 32, 48, 0.8) 0%, rgba(45, 50, 80, 0.8) 100%)',
    'accent_blue': '#4c7fff',
    'accent_purple': '#9f7aea',
    'accent_gradient': 'linear-gradient(90deg, #4c7fff 0%, #9f7aea 100%)'
}

# Custom card component for consistent UI
def create_card(title, content, id=None):
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(title, className="mb-0"),
                className="bg-transparent border-bottom border-dark"
            ),
            dbc.CardBody(content)
        ],
        id=id,
        className="shadow-sm mb-4"
    )

# Sidebar for navigation
sidebar = html.Div(
    [
        html.Div([
            html.I(className="fas fa-robot fa-2x text-primary"),
            html.H4("AgenDev", className="mt-2 mb-4")
        ], className="text-center mb-4"),
        dbc.Nav([
            dbc.NavLink(
                [html.I(className="fas fa-home me-2"), "Overview"],
                href="#",
                id="nav-overview", 
                active=True
            ),
            dbc.NavLink(
                [html.I(className="fas fa-tasks me-2"), "Tasks & Epics"],
                href="#",
                id="nav-tasks"
            ),
            dbc.NavLink(
                [html.I(className="fas fa-code me-2"), "Implementation"],
                href="#",
                id="nav-implementation"
            ),
            dbc.NavLink(
                [html.I(className="fas fa-cog me-2"), "Settings"],
                href="#",
                id="nav-settings"
            ),
        ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Div([
            html.P("Project Status"),
            dbc.Badge(
                agendev.project_state.value.capitalize(), 
                color="success", 
                className="mb-3"
            ),
        ]),
    ],
    style={
        "background": "linear-gradient(180deg, #1a1f38 0%, #2d305c 100%)",
        "padding": "20px",
        "borderRadius": "12px",
        "height": "100%",
    },
    className="shadow",
)

# Create pages
def create_overview_page():
    """Generate the overview page content"""
    project_status = agendev.get_project_status()
    progress = project_status.get('progress', {}).get('percentage', 0)
    
    # Status card
    status_card = create_card(
        "Project Status",
        [
            dbc.Progress(value=progress, color="success", className="mb-3"),
            html.Div([
                html.Span("Status: ", className="text-secondary"),
                html.Span(project_status.get('state', 'Not started').capitalize()),
            ], className="mb-2"),
            html.Div([
                html.Span("Progress: ", className="text-secondary"),
                html.Span(f"{progress:.1f}%"),
            ], className="mb-2"),
        ]
    )
    
    # Tasks overview
    tasks_card = create_card(
        "Tasks Overview",
        [
            html.Div([
                html.I(className="fas fa-tasks me-2 text-primary"),
                html.Span("Total Tasks: ", className="text-secondary"),
                html.Span(str(project_status.get('tasks', {}).get('total', 0))),
            ], className="mb-2"),
            html.Div([
                html.I(className="fas fa-check-circle me-2 text-success"),
                html.Span("Completed: ", className="text-secondary"),
                html.Span(str(project_status.get('tasks', {}).get('by_status', {}).get(TaskStatus.COMPLETED.value, 0))),
            ], className="mb-2"),
        ]
    )
    
    # Summary card
    summary_card = create_card(
        "Project Summary",
        [
            html.P(agendev.summarize_progress(voice_summary=False).replace('\n', '').strip() or "No project summary available."),
            dbc.Button(
                [html.I(className="fas fa-volume-up me-2"), "Generate Voice Summary"],
                id="btn-voice-summary",
                color="primary",
                className="mt-2",
            ),
            html.Div(id="voice-summary-output", className="mt-2")
        ]
    )
    
    return dbc.Container([
        html.H2("Project Overview", className="mb-4"),
        dbc.Row([
            dbc.Col(status_card, md=6),
            dbc.Col(tasks_card, md=6),
        ]),
        dbc.Row([
            dbc.Col(summary_card, md=12),
        ]),
    ])

def create_tasks_page():
    """Generate the tasks page content"""
    # Create form for adding a new task
    add_task_form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Title"),
                dbc.Input(id="task-title", type="text", placeholder="Task title"),
            ], md=6),
            dbc.Col([
                dbc.Label("Type"),
                dcc.Dropdown(
                    id="task-type",
                    options=[
                        {"label": task_type.value.capitalize(), "value": task_type.value}
                        for task_type in TaskType
                    ],
                    value=TaskType.IMPLEMENTATION.value,
                ),
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Description"),
                dbc.Textarea(id="task-description", placeholder="Task description"),
            ]),
        ], className="mt-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Priority"),
                dcc.Dropdown(
                    id="task-priority",
                    options=[
                        {"label": priority.value.capitalize(), "value": priority.value}
                        for priority in TaskPriority
                    ],
                    value=TaskPriority.MEDIUM.value,
                ),
            ], md=6),
            dbc.Col([
                dbc.Label("Duration (hours)"),
                dbc.Input(id="task-duration", type="number", value=1.0, min=0.1, step=0.1),
            ], md=6),
        ], className="mt-3"),
        dbc.Button(
            [html.I(className="fas fa-plus me-2"), "Create Task"], 
            id="btn-create-task", 
            color="primary",
            className="mt-3",
        ),
        html.Div(id="task-creation-output", className="mt-3")
    ])
    
    # Extract task data for the table
    tasks_data = []
    for task_id, task in agendev.task_graph.tasks.items():
        tasks_data.append({
            "id": str(task_id),
            "title": task.title,
            "status": task.status.value,
            "type": task.task_type.value,
            "duration": f"{task.estimated_duration_hours:.1f}h",
        })
    
    # Create tasks table
    if tasks_data:
        tasks_table = dbc.Table.from_dataframe(
            pd.DataFrame(tasks_data)[["title", "status", "type", "duration"]],
            striped=True,
            bordered=False,
            hover=True,
            className="table-dark",
        )
    else:
        tasks_table = html.Div(
            html.P("No tasks found. Create your first task using the form below."),
            className="text-center p-5"
        )
    
    return dbc.Container([
        html.H2("Tasks & Epics Management", className="mb-4"),
        dbc.Row([
            dbc.Col([
                create_card("Tasks", tasks_table),
                create_card("Add New Task", add_task_form),
            ]),
        ]),
    ])

def create_implementation_page():
    """Generate the implementation page content"""
    # Get current plan
    current_plan = agendev.planning_history.get_current_plan()
    
    # Create task selection dropdown if we have a plan
    task_options = []
    
    if current_plan:
        for task_id in current_plan.task_sequence:
            if isinstance(task_id, UUID) and task_id in agendev.task_graph.tasks:
                task = agendev.task_graph.tasks[task_id]
                # Only show tasks that are either planned or in progress
                if task.status in [TaskStatus.PLANNED, TaskStatus.IN_PROGRESS]:
                    task_options.append({
                        "label": task.title,
                        "value": str(task_id)
                    })
    
    # Implementation form
    implementation_form = html.Div([
        dbc.Label("Select Task to Implement"),
        dcc.Dropdown(
            id="implementation-task",
            options=task_options,
            placeholder="Select a task...",
            className="mb-3",
        ) if task_options else html.P("No tasks available for implementation. Generate a plan first."),
        dbc.Button(
            [html.I(className="fas fa-code me-2"), "Generate Implementation"], 
            id="btn-generate-implementation",
            color="primary",
            className="mt-2",
            disabled=not task_options,
        ),
        html.Div(id="implementation-output", className="mt-3"),
    ])
    
    # Find implemented files to display
    implementations = []
    for task_id, task in agendev.task_graph.tasks.items():
        if task.status == TaskStatus.COMPLETED and task.artifact_paths:
            for path in task.artifact_paths:
                try:
                    file_path = resolve_path(path)
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            content = f.read()
                        implementations.append({
                            "task_id": str(task_id),
                            "task_title": task.title,
                            "file_path": str(path),
                            "file_name": Path(path).name,
                            "content": content,
                        })
                except Exception:
                    pass
    
    # Code viewer
    code_content = ""
    if implementations:
        code_content = html.Div([
            html.H6(implementations[0]["file_name"], className="mb-3"),
            dbc.Card(
                dbc.CardBody(
                    dcc.Markdown(
                        f"```python\n{implementations[0]['content']}\n```",
                        className="mb-0",
                    )
                ),
                className="bg-dark"
            )
        ])
    else:
        code_content = html.P("No implementations available. Generate one using the form.")
    
    return dbc.Container([
        html.H2("Implementation", className="mb-4"),
        dbc.Row([
            dbc.Col(create_card("Implement Task", implementation_form), md=4),
            dbc.Col(create_card("Implementation Code", html.Div(code_content, id="code-viewer")), md=8),
        ]),
    ])

def create_settings_page():
    """Generate the settings page content"""
    # Read current settings from config
    current_settings = {
        "project_name": agendev.config.project_name,
        "llm_base_url": agendev.config.llm_base_url,
        "tts_base_url": agendev.config.tts_base_url,
        "notifications_enabled": agendev.config.notifications_enabled,
    }
    
    # Project settings form
    project_settings_form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Project Name"),
                dbc.Input(
                    id="setting-project-name",
                    value=current_settings["project_name"],
                    className="mb-3",
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Notifications"),
                dbc.Switch(
                    id="setting-notifications",
                    value=current_settings["notifications_enabled"],
                    className="mb-3",
                ),
            ]),
        ]),
        dbc.Button(
            [html.I(className="fas fa-save me-2"), "Save Settings"], 
            id="btn-save-settings",
            color="primary",
            className="mt-2",
        ),
        html.Div(id="settings-output", className="mt-3"),
    ])
    
    # API settings
    api_settings_form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("LLM API URL"),
                dbc.Input(
                    id="setting-llm-url",
                    value=current_settings["llm_base_url"],
                    className="mb-3",
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("TTS API URL"),
                dbc.Input(
                    id="setting-tts-url",
                    value=current_settings["tts_base_url"],
                    className="mb-3",
                ),
            ]),
        ]),
        dbc.Button(
            [html.I(className="fas fa-save me-2"), "Save API Settings"], 
            id="btn-save-api-settings",
            color="primary",
            className="mt-2",
        ),
        html.Div(id="api-settings-output", className="mt-3"),
    ])
    
    return dbc.Container([
        html.H2("Settings", className="mb-4"),
        dbc.Row([
            dbc.Col(create_card("Project Settings", project_settings_form), md=6),
            dbc.Col(create_card("API Settings", api_settings_form), md=6),
        ]),
    ])

# Content container that updates based on active page
content = html.Div(id="page-content", children=create_overview_page())

# App layout
app.layout = html.Div(
    [
        dbc.Row([
            dbc.Col(sidebar, width=2, className="sidebar"),
            dbc.Col([
                dbc.Navbar(
                    dbc.Container(
                        [
                            html.A(
                                dbc.Row([
                                    dbc.Col(html.I(className="fas fa-robot fa-lg")),
                                    dbc.Col(dbc.NavbarBrand("AgenDev Dashboard", className="ms-2")),
                                ], align="center"),
                                href="/",
                            ),
                        ],
                        fluid=True,
                    ),
                    color="dark",
                    dark=True,
                    className="mb-4",
                ),
                content,
            ], width=10),
        ], className="h-100 mx-0"),
        dcc.Store(id="active-page", data="overview"),
    ],
    style={
        "background": styles['background'],
        "minHeight": "100vh",
        "color": "white",
    }
)

# Callbacks for navigation
@callback(
    Output("active-page", "data"),
    [
        Input("nav-overview", "n_clicks"),
        Input("nav-tasks", "n_clicks"),
        Input("nav-implementation", "n_clicks"),
        Input("nav-settings", "n_clicks"),
    ],
    [State("active-page", "data")],
    prevent_initial_call=True,
)
def update_active_page(overview_clicks, tasks_clicks, implementation_clicks, settings_clicks, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    page_map = {
        "nav-overview": "overview",
        "nav-tasks": "tasks",
        "nav-implementation": "implementation",
        "nav-settings": "settings",
    }
    return page_map.get(button_id, current)

@callback(
    [
        Output("nav-overview", "active"),
        Output("nav-tasks", "active"),
        Output("nav-implementation", "active"),
        Output("nav-settings", "active"),
    ],
    Input("active-page", "data"),
)
def update_nav_active(active_page):
    return (
        active_page == "overview",
        active_page == "tasks",
        active_page == "implementation", 
        active_page == "settings",
    )

@callback(
    Output("page-content", "children"),
    Input("active-page", "data"),
)
def render_page_content(active_page):
    if active_page == "overview":
        return create_overview_page()
    elif active_page == "tasks":
        return create_tasks_page()
    elif active_page == "implementation":
        return create_implementation_page()
    elif active_page == "settings":
        return create_settings_page()
    else:
        return create_overview_page()

# Task creation callback
@callback(
    Output("task-creation-output", "children"),
    Input("btn-create-task", "n_clicks"),
    [
        State("task-title", "value"),
        State("task-description", "value"),
        State("task-type", "value"),
        State("task-priority", "value"),
        State("task-duration", "value"),
    ],
    prevent_initial_call=True,
)
def handle_create_task(n_clicks, title, description, task_type, priority, duration):
    if not title or not description:
        return dbc.Alert("Title and description are required", color="danger")
    
    try:
        # Convert values to proper types
        task_type_enum = TaskType(task_type)
        priority_enum = TaskPriority(priority)
        risk_enum = TaskRisk.MEDIUM  # Default risk
        
        # Create the task
        task_id = agendev.create_task(
            title=title,
            description=description,
            task_type=task_type_enum,
            priority=priority_enum,
            risk=risk_enum,
            estimated_duration_hours=float(duration),
        )
        
        return dbc.Alert(f"Task '{title}' created successfully", color="success")
    except Exception as e:
        return dbc.Alert(f"Error creating task: {str(e)}", color="danger")

# Implementation callback
@callback(
    Output("implementation-output", "children"),
    Input("btn-generate-implementation", "n_clicks"),
    [State("implementation-task", "value")],
    prevent_initial_call=True,
)
def handle_implementation(n_clicks, task_id):
    if not task_id:
        return dbc.Alert("Please select a task to implement", color="danger")
    
    try:
        # Implement the task
        result = agendev.implement_task(UUID(task_id))
        
        if "error" in result:
            return dbc.Alert(f"Error: {result['error']}", color="danger")
            
        return dbc.Alert(f"Task '{result['title']}' implemented successfully", color="success")
    except Exception as e:
        return dbc.Alert(f"Error implementing task: {str(e)}", color="danger")

# Voice summary callback
@callback(
    Output("voice-summary-output", "children"),
    Input("btn-voice-summary", "n_clicks"),
    prevent_initial_call=True,
)
def handle_voice_summary(n_clicks):
    try:
        # Generate voice summary
        summary = agendev.summarize_progress(voice_summary=True)
        
        return dbc.Alert("Voice summary generated successfully", color="success")
    except Exception as e:
        return dbc.Alert(f"Error generating voice summary: {str(e)}", color="danger")

# Settings callbacks
@callback(
    Output("settings-output", "children"),
    Input("btn-save-settings", "n_clicks"),
    [
        State("setting-project-name", "value"),
        State("setting-notifications", "value"),
    ],
    prevent_initial_call=True,
)
def handle_save_settings(n_clicks, project_name, notifications_enabled):
    if not project_name:
        return dbc.Alert("Project name is required", color="danger")
    
    try:
        # Update config
        agendev.config.project_name = project_name
        agendev.config.notifications_enabled = bool(notifications_enabled)
        
        # Save project state to reflect changes
        agendev._save_project_state()
        
        return dbc.Alert("Settings saved successfully", color="success")
    except Exception as e:
        return dbc.Alert(f"Error saving settings: {str(e)}", color="danger")

@callback(
    Output("api-settings-output", "children"),
    Input("btn-save-api-settings", "n_clicks"),
    [
        State("setting-llm-url", "value"),
        State("setting-tts-url", "value"),
    ],
    prevent_initial_call=True,
)
def handle_save_api_settings(n_clicks, llm_url, tts_url):
    if not llm_url or not tts_url:
        return dbc.Alert("All URL fields are required", color="danger")
    
    try:
        # Update config
        agendev.config.llm_base_url = llm_url
        agendev.config.tts_base_url = tts_url
        
        # Update clients
        if hasattr(agendev, 'llm'):
            agendev.llm.llm_client = LLMClient(base_url=llm_url)
        
        if hasattr(agendev, 'notification_manager') and agendev.notification_manager:
            agendev.notification_manager.tts_client = TTSClient(base_url=tts_url)
        
        return dbc.Alert("API settings saved successfully", color="success")
    except Exception as e:
        return dbc.Alert(f"Error saving API settings: {str(e)}", color="danger")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)