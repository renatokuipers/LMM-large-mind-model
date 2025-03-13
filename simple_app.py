import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import time
import os
import asyncio
import uuid
from datetime import datetime
import threading
import diskcache
import traceback
from concurrent.futures import ThreadPoolExecutor
import atexit
import sys
import signal

# Set up global executor for background tasks
background_executor = ThreadPoolExecutor(max_workers=2)

# Register cleanup on application exit
def cleanup_executor():
    print("Shutting down executor...")
    try:
        background_executor.shutdown(wait=False)
    except:
        pass

atexit.register(cleanup_executor)

# Handle signals gracefully
def signal_handler(sig, frame):
    print("Intercepted signal, cleaning up...")
    cleanup_executor()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

# Set up diskcache for callbacks
cache = diskcache.Cache("./cache")

# Clear the cache on startup to prevent stale callback issues
def clear_cache():
    try:
        cache.clear()
        print("Cleared callback cache")
    except Exception as e:
        print(f"Error clearing cache: {e}")

# Execute cache clearing
clear_cache()

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    url_base_pathname='/',
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

# Simplified app state with a centralized pattern
app_state = {
    "view": "landing",
    "initial_prompt": "",
    "project_id": None,
    "tasks": []
}

# App layout
landing_page = html.Div(
    id="landing-page",
    className="landing-page",
    style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "height": "100vh"},
    children=[
        html.Div("AgenDev", style={"fontSize": "2.5rem", "fontWeight": "bold", "color": "#61dafb", "marginBottom": "30px"}),
        html.Div(
            "An Intelligent Agentic Development System", 
            style={"fontSize": "1rem", "color": "#888", "marginBottom": "30px", "textAlign": "center"}
        ),
        html.Div(
            style={"width": "80%", "maxWidth": "800px", "padding": "20px", "backgroundColor": "#333", "borderRadius": "8px", "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"},
            children=[
                html.H2("What would you like to develop today?", style={"fontSize": "1.5rem", "marginBottom": "20px", "textAlign": "center"}),
                dcc.Textarea(
                    id="initial-prompt",
                    placeholder="Describe your project or what you'd like help with...",
                    style={
                        "width": "100%",
                        "height": "120px",
                        "borderRadius": "4px",
                        "padding": "10px",
                        "marginBottom": "15px",
                        "backgroundColor": "#2a2a2a",
                        "color": "#fff",
                        "border": "1px solid #444"
                    }
                ),
                html.Button(
                    "Submit",
                    id="submit-button",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "4px",
                        "backgroundColor": "#61dafb",
                        "color": "#000",
                        "border": "none",
                        "cursor": "pointer",
                        "fontWeight": "bold"
                    }
                )
            ]
        )
    ]
)

# Main dashboard that shows after submitting a prompt
main_dashboard = html.Div(
    id="main-dashboard",
    style={"display": "none"},
    children=[
        html.Div(
            style={"padding": "20px", "backgroundColor": "#1E1E1E"},
            children=[
                html.H2(id="project-title", children="Project Title", style={"marginBottom": "15px"}),
                html.Div(
                    id="status-message", 
                    style={"padding": "10px", "marginBottom": "20px", "backgroundColor": "#333", "borderRadius": "5px"},
                    children="Initializing your project..."
                ),
                html.Div(id="chat-container", children=[])
            ]
        )
    ]
)

# Create the app layout
app.layout = html.Div([
    dcc.Store(id='app-state-store', data=app_state),
    landing_page,
    main_dashboard,
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True)
])

# Simplified callback for transitioning from landing to main view
@app.callback(
    [Output('app-state-store', 'data'),
     Output('landing-page', 'style'),
     Output('main-dashboard', 'style'),
     Output('project-title', 'children'),
     Output('status-message', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('initial-prompt', 'value'),
     State('app-state-store', 'data')],
    prevent_initial_call=True
)
def handle_submit(n_clicks, prompt_value, current_state):
    if not n_clicks or not prompt_value:
        raise PreventUpdate
    
    # Update the app state
    current_state['view'] = 'main'
    current_state['initial_prompt'] = prompt_value
    current_state['project_id'] = str(uuid.uuid4())
    
    # Hide landing page, show main dashboard
    landing_style = {"display": "none"}
    main_style = {"display": "block"}
    
    # Set project title and status message
    project_title = f"Project: {prompt_value[:30]}..." if len(prompt_value) > 30 else f"Project: {prompt_value}"
    status_message = f"Processing your request: {prompt_value}"
    
    return current_state, landing_style, main_style, project_title, status_message

# Callback to update chat container periodically
@app.callback(
    Output('chat-container', 'children'),
    [Input('interval-component', 'n_intervals')],
    [State('app-state-store', 'data')],
    prevent_initial_call=True
)
def update_chat(n_intervals, current_state):
    if current_state['view'] != 'main':
        raise PreventUpdate
    
    # Create a chat message showing progress
    progress = min(n_intervals * 10, 100)
    
    chat_messages = [
        html.Div(
            style={"padding": "15px", "marginBottom": "10px", "backgroundColor": "#2A2A2A", "borderRadius": "5px"},
            children=[
                html.Div(f"Processing your request - {progress}% complete"),
                html.Div(
                    style={"height": "10px", "backgroundColor": "#444", "borderRadius": "5px", "marginTop": "10px"},
                    children=[
                        html.Div(
                            style={
                                "height": "100%", 
                                "width": f"{progress}%", 
                                "backgroundColor": "#61dafb",
                                "borderRadius": "5px"
                            }
                        )
                    ]
                )
            ]
        )
    ]
    
    # Add some sample agent messages
    if n_intervals >= 3:
        chat_messages.append(
            html.Div(
                style={"padding": "15px", "marginBottom": "10px", "backgroundColor": "#333", "borderRadius": "5px", "borderLeft": "4px solid #61dafb"},
                children=[
                    html.Div(style={"fontWeight": "bold", "marginBottom": "5px"}, children="Planning Agent"),
                    html.Div("Analyzing your requirements and creating project structure...")
                ]
            )
        )
    
    if n_intervals >= 6:
        chat_messages.append(
            html.Div(
                style={"padding": "15px", "marginBottom": "10px", "backgroundColor": "#333", "borderRadius": "5px", "borderLeft": "4px solid #28a745"},
                children=[
                    html.Div(style={"fontWeight": "bold", "marginBottom": "5px"}, children="Code Agent"),
                    html.Div("Setting up development environment and creating initial files...")
                ]
            )
        )
    
    return chat_messages

# Enable the interval when transitioning to main view
@app.callback(
    Output('interval-component', 'disabled'),
    [Input('app-state-store', 'data')],
    prevent_initial_call=True
)
def toggle_interval(current_state):
    return current_state['view'] != 'main'

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0') 