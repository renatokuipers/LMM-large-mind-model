import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
import json
import time
from datetime import datetime
from pathlib import Path
import markdown
from typing import Dict, List, Optional, Any, Union

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

# App state store
app_state = dcc.Store(
    id='app-state',
    data={
        "view": "landing",
        "initial_prompt": "",
        "current_task_index": 0,
        "is_live_mode": True
    }
)

# Custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AgenDev - Intelligent Agentic Development System</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #1e1e1e;
                --secondary-color: #2a2a2a;
                --tertiary-color: #333;
                --text-color: #fff;
                --text-secondary: #ccc;
                --text-muted: #888;
                --accent-color: #61dafb;
                --success-color: #00ff00;
                --warning-color: #ffc107;
                --danger-color: #dc3545;
                --purple-accent: #800080;
            }
            
            body {
                background-color: var(--primary-color);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
                margin: 0;
                padding: 0;
                height: 100vh;
            }
            
            .landing-page {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
            }
            
            .main-container {
                display: flex;
                height: 100vh;
                width: 100%;
                overflow: hidden;
            }
            
            .chat-container {
                width: 50%;
                height: 100%;
                overflow-y: auto;
                padding: 20px;
                background-color: var(--primary-color);
                border-right: 1px solid var(--tertiary-color);
            }
            
            .view-container {
                width: 50%;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                background-color: var(--primary-color);
            }
            
            .view-header {
                background-color: var(--tertiary-color);
                padding: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .view-content {
                flex-grow: 1;
                overflow: auto;
                padding: 0;
                background-color: var(--secondary-color);
            }
            
            .view-controls {
                background-color: var(--tertiary-color);
                padding: 10px;
                display: flex;
                justify-content: space-between;
            }
            
            .chat-message {
                margin-bottom: 20px;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .system-message {
                padding: 15px;
                background-color: var(--tertiary-color);
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .user-message {
                padding: 15px;
                background-color: var(--secondary-color);
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .collapsible-header {
                display: flex;
                align-items: center;
                padding: 10px;
                background-color: var(--tertiary-color);
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            
            .collapsible-header:hover {
                background-color: #444;
            }
            
            .collapsible-content {
                padding: 10px;
                background-color: var(--secondary-color);
                border-radius: 4px;
                margin-bottom: 15px;
                margin-left: 15px;
                border-left: 2px solid var(--accent-color);
            }
            
            .command-element {
                background-color: var(--secondary-color);
                padding: 8px 12px;
                border-radius: 4px;
                margin: 5px 0;
                font-family: 'Consolas', 'Courier New', monospace;
                border-left: 3px solid var(--accent-color);
            }
            
            .status-element {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            
            .status-icon {
                margin-right: 10px;
            }
            
            .terminal-view {
                background-color: #1e1e1e;
                color: #ddd;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 10px;
                height: 100%;
                overflow: auto;
                white-space: pre-wrap;
            }
            
            .editor-view {
                background-color: #1e1e1e;
                height: 100%;
                overflow: auto;
            }
            
            .editor-header {
                background-color: #2d2d2d;
                padding: 5px 10px;
                border-bottom: 1px solid #444;
                display: flex;
                justify-content: space-between;
            }
            
            .editor-content {
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                color: #ddd;
                min-height: calc(100% - 40px);
            }
            
            .browser-view {
                background-color: #fff;
                height: 100%;
                overflow: auto;
            }
            
            .file-path {
                font-family: 'Consolas', 'Courier New', monospace;
                color: #888;
                font-size: 0.85em;
                margin-bottom: 5px;
            }
            
            .function-tag {
                background-color: var(--accent-color);
                color: #000;
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 5px;
                font-size: 0.8em;
            }
            
            .status-tag {
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 5px;
                font-size: 0.8em;
            }
            
            .status-tag.success {
                background-color: var(--success-color);
                color: #000;
            }
            
            .status-tag.in-progress {
                background-color: var(--warning-color);
                color: #000;
            }
            
            .status-tag.error {
                background-color: var(--danger-color);
                color: #fff;
            }
            
            .progress-controls {
                display: flex;
                align-items: center;
            }
            
            .time-indicator {
                font-size: 0.8em;
                color: #888;
                margin-left: 10px;
            }
            
            .btn-control {
                background: none;
                border: none;
                color: #888;
                font-size: 1em;
                cursor: pointer;
                padding: 5px;
                transition: color 0.2s;
            }
            
            .btn-control:hover {
                color: var(--accent-color);
            }
            
            .btn-control.active {
                color: var(--success-color);
            }
            
            .code-content {
                border-radius: 4px;
                background-color: #2d2d2d;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                overflow-x: auto;
            }
            
            .input-prompt {
                width: 80%;
                max-width: 800px;
                padding: 20px;
                background-color: var(--tertiary-color);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .input-heading {
                font-size: 1.5rem;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .brand-logo {
                margin-bottom: 30px;
                font-size: 2.5rem;
                font-weight: bold;
                color: var(--accent-color);
            }
            
            .brand-slogan {
                font-size: 1rem;
                color: var(--text-muted);
                margin-bottom: 30px;
                text-align: center;
            }
            
            .todo-markdown h1 {
                color: var(--accent-color);
                font-size: 1.8rem;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            
            .todo-markdown h2 {
                color: var(--accent-color);
                font-size: 1.4rem;
                margin-top: 0.8rem;
                margin-bottom: 0.4rem;
            }
            
            .todo-markdown h3 {
                color: var(--accent-color);
                font-size: 1.2rem;
                margin-top: 0.6rem;
                margin-bottom: 0.3rem;
            }
            
            .todo-markdown ul {
                padding-left: 20px;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            
            .todo-markdown li {
                margin-bottom: 0.3rem;
            }
            
            .todo-markdown input[type="checkbox"] {
                margin-right: 0.5rem;
            }
            
            .todo-markdown input[type="checkbox"]:checked + span {
                text-decoration: line-through;
                opacity: 0.7;
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

# ========== COMPONENT FUNCTIONS ==========

def create_terminal_view(content: str) -> html.Div:
    """Create a terminal view component"""
    return html.Div(
        className="terminal-view",
        children=[html.Pre(content)]
    )

def create_editor_view(filename: str, content: str, language: str = "text") -> html.Div:
    """Create an editor view component with syntax highlighting"""
    return html.Div(
        className="editor-view",
        children=[
            html.Div(
                className="editor-header",
                children=[
                    html.Div(filename),
                    html.Div([
                        html.Button("Diff", className="btn-control"),
                        html.Button("Original", className="btn-control"),
                        html.Button("Modified", className="btn-control", style={"color": "#fff"}),
                    ])
                ]
            ),
            html.Pre(
                content,
                className="editor-content",
                style={"whiteSpace": "pre-wrap"}
            )
        ]
    )

def create_collapsible_section(id_prefix: str, header_content: Union[html.Div, str], 
                              content: List, is_open: bool = True) -> html.Div:
    """Create a collapsible section component"""
    return html.Div([
        html.Div(
            className="collapsible-header",
            id=f"{id_prefix}-header",
            children=[
                html.I(
                    className="fas fa-chevron-down mr-2",
                    style={"marginRight": "10px"}
                ),
                header_content
            ]
        ),
        html.Div(
            id=f"{id_prefix}-content",
            className="collapsible-content",
            style={"display": "block" if is_open else "none"},
            children=content
        )
    ])

def create_command_element(command: str, status: str = "completed") -> html.Div:
    """Create a command execution element"""
    icon_class = ("fas fa-check-circle text-success" if status == "completed" 
                else "fas fa-spinner fa-spin text-warning")
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span("Executing command", style={"marginRight": "10px"}),
            html.Code(command, className="command-element")
        ]
    )

def create_file_operation(operation: str, filepath: str, status: str = "completed") -> html.Div:
    """Create a file operation status element"""
    icon_class = ("fas fa-check-circle text-success" if status == "completed" 
                else "fas fa-spinner fa-spin text-warning")
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span(f"{operation} file", style={"marginRight": "10px"}),
            html.Code(filepath, className="file-path")
        ]
    )

def render_markdown(markdown_text: str) -> dcc.Markdown:
    """Render markdown content with custom styling"""
    # Process checkboxes with custom rendering
    lines = markdown_text.split('\n')
    for i, line in enumerate(lines):
        if '- [ ]' in line:
            lines[i] = line.replace('- [ ]', '- <input type="checkbox"><span>')
            lines[i] += '</span>'
        elif '- [x]' in line:
            lines[i] = line.replace('- [x]', '- <input type="checkbox" checked><span>')
            lines[i] += '</span>'
    
    processed_markdown = '\n'.join(lines)
    
    # Return Markdown component with proper HTML handling
    return dcc.Markdown(
        processed_markdown,
        className="todo-markdown",
        dangerously_allow_html=True
    )

# ========== LAYOUT COMPONENTS ==========

# Landing page layout with centered input
landing_page = html.Div(
    id="landing-page",
    className="landing-page",
    children=[
        html.Div(className="brand-logo", children=["AgenDev"]),
        html.Div(
            className="brand-slogan", 
            children=["An Intelligent Agentic Development System"]
        ),
        html.Div(
            className="input-prompt",
            children=[
                html.H2("What would you like to develop today?", className="input-heading"),
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

# Main split view layout
main_view = html.Div(
    id="main-container",
    className="main-container",
    style={"display": "none"},  # Hidden initially
    children=[
        # Left side - Chat interface
        html.Div(
            className="chat-container",
            id="chat-container",
            children=[
                # Header with current task
                html.Div(
                    className="system-message",
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "marginBottom": "15px"},
                            children=[
                                html.I(className="fas fa-robot", style={"fontSize": "24px", "marginRight": "10px", "color": "#61dafb"}),
                                html.Span("AgenDev", style={"fontSize": "24px", "fontWeight": "bold"})
                            ]
                        ),
                        html.H3(id="project-title", children="Project Development", style={"margin": "0 0 10px 0"})
                    ]
                ),
                
                # Todo.md display area
                html.Div(
                    id="todo-display",
                    className="system-message",
                    style={"marginTop": "20px", "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "marginBottom": "15px"},
                            children=[
                                html.I(className="fas fa-tasks", style={"fontSize": "18px", "marginRight": "10px", "color": "#61dafb"}),
                                html.Span("Project Tasks", style={"fontSize": "18px", "fontWeight": "bold"})
                            ]
                        ),
                        html.Div(id="todo-content")
                    ]
                ),
                
                # Task sections will be dynamically generated
                html.Div(id="task-sections")
            ]
        ),
        
        # Right side - Dynamic view (Terminal, Editor, Browser)
        html.Div(
            className="view-container",
            children=[
                # Header
                html.Div(
                    className="view-header",
                    children=[
                        html.Div("AgenDev's Computer"),
                        html.Button(
                            html.I(className="fas fa-expand"),
                            className="btn-control",
                            id="expand-view"
                        )
                    ]
                ),
                
                # View type indicator
                html.Div(
                    id="view-type-indicator",
                    style={
                        "padding": "5px 10px",
                        "backgroundColor": "#2d2d2d",
                        "borderBottom": "1px solid #444",
                        "display": "flex",
                        "alignItems": "center"
                    },
                    children=[
                        html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
                        html.Span(id="current-view-type", children="Editor"),
                        html.Div(
                            id="file-operation-indicator",
                            style={
                                "marginLeft": "20px",
                                "display": "flex",
                                "alignItems": "center",
                                "color": "#888",
                                "fontSize": "0.85em"
                            },
                            children=[
                                html.Span(id="operation-type", children="Creating file"),
                                html.Code(
                                    id="current-file-path",
                                    children="project/file.txt",
                                    style={
                                        "marginLeft": "5px",
                                        "backgroundColor": "transparent",
                                        "padding": "0"
                                    }
                                )
                            ]
                        )
                    ]
                ),
                
                # Content area (can be terminal, editor, or browser)
                html.Div(
                    className="view-content",
                    id="view-content",
                    children=[]  # Will be dynamically populated
                ),
                
                # Replay Controls
                html.Div(
                    className="view-controls",
                    children=[
                        html.Div(
                            className="progress-controls",
                            children=[
                                html.Button(
                                    html.I(className="fas fa-step-backward"),
                                    className="btn-control",
                                    id="playback-backward",
                                    title="Previous step"
                                ),
                                html.Button(
                                    html.I(id="play-icon", className="fas fa-play"),
                                    className="btn-control",
                                    id="playback-play",
                                    title="Play/Pause"
                                ),
                                html.Button(
                                    html.I(className="fas fa-step-forward"),
                                    className="btn-control",
                                    id="playback-forward",
                                    title="Next step"
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="playback-slider",
                                        min=0,
                                        max=100,
                                        value=0,
                                        updatemode="drag",
                                        marks=None,
                                        tooltip={"always_visible": False, "placement": "bottom"},
                                        className="timeline-slider"
                                    ),
                                    style={"width": "300px", "marginLeft": "10px", "marginRight": "10px"}
                                ),
                                html.Button(
                                    html.I(className="fas fa-bolt"),
                                    className="btn-control active",
                                    id="live-button",
                                    title="Live mode"
                                )
                            ]
                        ),
                        html.Div(
                            className="status-indicator",
                            children=[
                                html.Span(
                                    html.I(id="task-status-icon", className="fas fa-spinner fa-spin"),
                                    id="task-status-tag",
                                    className="status-tag in-progress",
                                    style={"marginRight": "5px"}
                                ),
                                html.Span(id="current-task-text", children="Setting up project environment"),
                                html.Span(id="task-progress", children="1/5", className="time-indicator")
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Store for playback data
playback_store = dcc.Store(
    id='playback-data',
    data={
        "total_steps": 0,
        "current_step": 0,
        "is_playing": False,
        "play_interval": 3,  # seconds between steps
        "steps": []
    }
)

# Store for todo.md content
todo_store = dcc.Store(
    id='todo-data',
    data={
        "content": "# Project Development\n## Setup Phase\n- [ ] Initialize project repository\n- [ ] Set up development environment\n## Implementation Phase\n- [ ] Implement core functionality\n- [ ] Create user interface\n## Testing Phase\n- [ ] Write unit tests\n- [ ] Perform integration testing"
    }
)

# Store for task sections
task_store = dcc.Store(
    id='task-data',
    data={
        "tasks": [
            {
                "id": "task1",
                "title": "Initialize project repository",
                "status": "in-progress",
                "content": [
                    html.P("Setting up the project repository and initial structure."),
                    create_command_element("git init", "completed"),
                    create_command_element("npm init -y", "in-progress")
                ]
            }
        ]
    }
)

# Interval for playback
playback_interval = dcc.Interval(
    id='playback-interval',
    interval=3000,  # 3 seconds between steps
    n_intervals=0,
    disabled=True
)

# Combine all components into the app layout
app.layout = html.Div([
    app_state,
    playback_store,
    todo_store,
    task_store,
    playback_interval,
    landing_page,
    main_view
])

# ========== CALLBACKS ==========

# Callback to transition from landing page to main view
@app.callback(
    [Output("app-state", "data"),
     Output("landing-page", "style"),
     Output("main-container", "style"),
     Output("project-title", "children"),
     Output("todo-data", "data"),
     Output("playback-data", "data")],
    [Input("submit-button", "n_clicks")],
    [State("initial-prompt", "value"),
     State("app-state", "data"),
     State("todo-data", "data"),
     State("playback-data", "data")],
    prevent_initial_call=True
)
def transition_to_main_view(n_clicks, prompt_value, current_state, todo_data, playback_data):
    if not n_clicks:
        raise PreventUpdate
    
    # Update state
    current_state["view"] = "main"
    current_state["initial_prompt"] = prompt_value
    
    # Generate a title based on the prompt
    title = "New Project"
    if prompt_value:
        # Simple algorithm to extract a title
        if len(prompt_value.split()) < 5:
            title = prompt_value.capitalize()
        else:
            words = prompt_value.split()
            title_words = words[:3]
            title = " ".join(word.capitalize() for word in title_words) + " Project"
    
    # Create a todo.md based on the prompt
    todo_content = f"# {title}\n\n## Setup Phase\n- [ ] Initialize project repository\n- [ ] Set up development environment\n\n## Implementation Phase\n- [ ] Design core architecture\n- [ ] Implement key features\n- [ ] Create user interface\n\n## Testing Phase\n- [ ] Write unit tests\n- [ ] Perform integration testing\n\n## Deployment Phase\n- [ ] Prepare deployment pipeline\n- [ ] Release version 1.0"
    
    todo_data["content"] = todo_content
    
    # Create sample playback steps
    playback_data["steps"] = [
        {
            "type": "terminal",
            "content": f"$ echo 'Initializing {title}'\nInitializing {title}\n$ mkdir {title.lower().replace(' ', '_')}\n$ cd {title.lower().replace(' ', '_')}\n$ git init\nInitialized empty Git repository in ./{title.lower().replace(' ', '_')}/.git/",
            "operation_type": "Setting up",
            "file_path": title.lower().replace(' ', '_')
        },
        {
            "type": "terminal",
            "content": f"$ npm init -y\nWrote to ./{title.lower().replace(' ', '_')}/package.json\n$ npm install --save-dev webpack webpack-cli\nAdded 214 packages, and audited 215 packages in 3s",
            "operation_type": "Installing",
            "file_path": "package.json"
        },
        {
            "type": "editor",
            "filename": "package.json",
            "content": '{\n  "name": "' + title.lower().replace(' ', '_') + '",\n  "version": "1.0.0",\n  "description": "' + title + '",\n  "main": "index.js",\n  "scripts": {\n    "test": "echo \\"Error: no test specified\\" && exit 1",\n    "start": "webpack --mode development",\n    "build": "webpack --mode production"\n  },\n  "keywords": [],\n  "author": "",\n  "license": "ISC",\n  "devDependencies": {\n    "webpack": "^5.75.0",\n    "webpack-cli": "^5.0.1"\n  }\n}',
            "operation_type": "Editing",
            "file_path": "package.json"
        },
        {
            "type": "editor",
            "filename": "README.md",
            "content": f"# {title}\n\nThis project was created with AgenDev, an Intelligent Agentic Development System.\n\n## Getting Started\n\n```bash\nnpm install\nnpm start\n```\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n\n## License\n\nMIT\n",
            "operation_type": "Creating",
            "file_path": "README.md"
        }
    ]
    
    playback_data["total_steps"] = len(playback_data["steps"])
    playback_data["current_step"] = 0
    
    # Hide landing page, show main container
    landing_style = {"display": "none"}
    main_style = {"display": "flex"}
    
    return current_state, landing_style, main_style, title, todo_data, playback_data

# Callback to render the todo.md content
@app.callback(
    Output("todo-content", "children"),
    [Input("todo-data", "data")],
    prevent_initial_call=True
)
def update_todo_content(todo_data):
    return render_markdown(todo_data["content"])

# Callback to render task sections
@app.callback(
    Output("task-sections", "children"),
    [Input("task-data", "data")],
    prevent_initial_call=True
)
def update_task_sections(task_data):
    task_sections = []
    
    for task in task_data.get("tasks", []):
        # Determine icon class based on status
        if task["status"] == "completed":
            icon_class = "fas fa-check-circle"
            icon_style = {"marginRight": "10px", "color": "#00ff00"}
        elif task["status"] == "in-progress":
            icon_class = "fas fa-spinner fa-spin"
            icon_style = {"marginRight": "10px", "color": "#ffc107"}
        else:
            icon_class = "fas fa-circle"
            icon_style = {"marginRight": "10px", "color": "#888"}
        
        # Create header with icon and title
        header_content = html.Div([
            html.I(className=icon_class, style=icon_style),
            html.Span(task["title"])
        ])
        
        # Create collapsible section
        section = create_collapsible_section(
            task["id"],
            header_content,
            task["content"],
            is_open=(task["status"] == "in-progress")
        )
        
        task_sections.append(section)
    
    return task_sections

# Callback for updating view content based on playback
@app.callback(
    [Output("view-content", "children"),
     Output("current-view-type", "children"),
     Output("operation-type", "children"),
     Output("current-file-path", "children")],
    [Input("playback-data", "data")],
    prevent_initial_call=True
)
def update_view_content(playback_data):
    if not playback_data or not playback_data.get("steps"):
        return [], "None", "", ""
    
    current_step = playback_data["current_step"]
    
    if current_step >= len(playback_data["steps"]):
        return [], "None", "", ""
    
    step_data = playback_data["steps"][current_step]
    view_type = step_data.get("type", "terminal")
    
    if view_type == "terminal":
        content = create_terminal_view(step_data.get("content", ""))
    elif view_type == "editor":
        content = create_editor_view(
            step_data.get("filename", "unnamed.txt"),
            step_data.get("content", ""),
            "text"
        )
    elif view_type == "browser":
        content = html.Iframe(
            src=step_data.get("url", "about:blank"),
            style={"width": "100%", "height": "100%", "border": "none"}
        )
    else:
        content = html.Div("No content available")
    
    # Update operation indicators
    operation_type = step_data.get("operation_type", "Working on")
    file_path = step_data.get("file_path", "")
    
    return [content], view_type.capitalize(), operation_type, file_path

# Callback for playback controls
@app.callback(
    [Output("playback-data", "data", allow_duplicate=True),
     Output("playback-interval", "disabled"),
     Output("play-icon", "className"),
     Output("live-button", "className"),
     Output("task-status-tag", "className"),
     Output("task-status-icon", "className"),
     Output("current-task-text", "children"),
     Output("task-progress", "children"),
     Output("playback-slider", "value")],
    [Input("playback-backward", "n_clicks"),
     Input("playback-play", "n_clicks"),
     Input("playback-forward", "n_clicks"),
     Input("live-button", "n_clicks"),
     Input("playback-interval", "n_intervals"),
     Input("playback-slider", "value")],
    [State("playback-data", "data"),
     State("app-state", "data")],
    prevent_initial_call=True
)
def control_playback(backward_clicks, play_clicks, forward_clicks, live_clicks, 
                    interval, slider_value, playback_data, app_state):
    # Get the component that triggered the callback
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    if not playback_data:
        raise PreventUpdate
    
    # Initialize values
    current_step = playback_data["current_step"]
    is_playing = playback_data["is_playing"]
    is_live = app_state.get("is_live_mode", True)
    total_steps = playback_data["total_steps"]
    
    # Handle different triggers
    if trigger_id == "playback-backward":
        current_step = max(0, current_step - 1)
        is_playing = False
        is_live = False
    
    elif trigger_id == "playback-play":
        is_playing = not is_playing
    
    elif trigger_id == "playback-forward":
        current_step = min(total_steps - 1, current_step + 1)
        if current_step == total_steps - 1:
            is_playing = False
            is_live = True
    
    elif trigger_id == "live-button":
        is_live = True
        current_step = total_steps - 1
        is_playing = False
    
    elif trigger_id == "playback-interval" and is_playing:
        current_step = min(total_steps - 1, current_step + 1)
        if current_step == total_steps - 1:
            is_playing = False
            is_live = True
    
    elif trigger_id == "playback-slider":
        # Calculate the step based on slider value
        current_step = round((slider_value / 100) * (total_steps - 1))
        is_playing = False
        is_live = (current_step == total_steps - 1)
    
    # Update playback data
    playback_data["current_step"] = current_step
    playback_data["is_playing"] = is_playing
    app_state["is_live_mode"] = is_live
    
    # Calculate slider value (0-100)
    if total_steps > 1:
        slider_value = (current_step / (total_steps - 1)) * 100
    else:
        slider_value = 0
    
    # Update task status
    status_class = "status-tag in-progress"
    icon_class = "fas fa-spinner fa-spin"
    if is_live and total_steps > 0:
        current_task = "Current task in progress..."
    else:
        step_index = min(current_step, total_steps - 1) if total_steps > 0 else 0
        step_data = playback_data["steps"][step_index] if playback_data["steps"] else {}
        current_task = f"Step {step_index + 1}: {step_data.get('operation_type', '')} {step_data.get('file_path', '')}"
    
    # Task progress indicator
    progress_text = f"{current_step + 1}/{total_steps}"
    
    # Button classes
    play_icon_class = "fas fa-pause" if is_playing else "fas fa-play"
    live_button_class = "btn-control active" if is_live else "btn-control"
    
    return (
        playback_data, 
        not is_playing,  # Interval is disabled when not playing
        play_icon_class,
        live_button_class,
        status_class,
        icon_class,
        current_task,
        progress_text,
        slider_value
    )

# Callbacks for collapsible sections
@app.callback(
    Output("task1-content", "style"),
    Input("task1-header", "n_clicks"),
    State("task1-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task1(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)