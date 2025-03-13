import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, ALLSMALLER
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import time
import os
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
import threading
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import traceback
from concurrent.futures import ThreadPoolExecutor
import atexit
import sys
import signal
from src.agendev.utils.fs_utils import get_workspace_root, ensure_workspace_structure

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

# Import agents and necessary modules for agentic functionality
from agendev.agents.planner_agent import PlannerAgent
from agendev.agents.code_agent import CodeAgent
from agendev.agents.integration_agent import IntegrationAgent
from agendev.agents.deployment_agent import DeploymentAgent
from agendev.agents.knowledge_agent import KnowledgeAgent
from agendev.utils.config import load_config
from agendev.utils.llm import LLMProvider
from agendev.llm_integration import LLMIntegration, LLMConfig

# Set up diskcache for long callbacks
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

long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    url_base_pathname='/',
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    long_callback_manager=long_callback_manager
)

# Clear the component registry to prevent callback hash issues
app._callback_list = []
app.callback_map = {}

# Load configuration
config = load_config()
# Use get_workspace_root to ensure projects are saved in the workspace directory
workspace_path = str(ensure_workspace_structure())
print(f"Using workspace directory: {workspace_path}")

# Initialize LLM provider and agents - with a single instance
llm_config = config.get("llm", {})
llm_provider = LLMProvider(llm_config)

# Initialize LLM integration for agents - ensure URL is complete
base_url = llm_config.get("endpoint", "http://192.168.2.12:1234")
# Ensure the URL is complete - append port if missing
if not base_url.endswith(":1234") and ":" not in base_url:
    base_url = f"{base_url}:1234"
print(f"Using LLM endpoint: {base_url}")
# Don't create the LLMProvider twice

model = llm_config.get("model", "qwen2.5-7b-instruct")
llm_integration_config = LLMConfig(
    model=model,
    temperature=0.7, 
    max_tokens=16384
)
llm_integration = LLMIntegration(
    base_url=base_url,
    config=llm_integration_config
)

# Initialize agents
planner_agent = PlannerAgent()
planner_agent.initialize(llm_integration)

code_agent = CodeAgent()
code_agent.initialize(llm_integration)

integration_agent = IntegrationAgent()
integration_agent.initialize(llm_integration)

deployment_agent = DeploymentAgent()
deployment_agent.initialize(llm_integration)

knowledge_agent = KnowledgeAgent()
knowledge_agent.initialize(llm_integration)

# Store for active projects
active_projects = {}

# Custom CSS for styling to match the screenshots
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
                --primary-color: #333;
                --secondary-color: #444;
                --text-color: #f8f9fa;
                --accent-color: #61dafb;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #ffc107;
            }
            
            body {
                background-color: #1a1a1a;
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
                background-color: #1a1a1a;
                border-right: 1px solid #333;
            }
            
            .view-container {
                width: 50%;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                background-color: #1a1a1a;
            }
            
            .view-header {
                background-color: #333;
                padding: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .view-content {
                flex-grow: 1;
                overflow: auto;
                padding: 0;
                background-color: #2a2a2a;
            }
            
            .view-controls {
                background-color: #333;
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
                background-color: #333;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .user-message {
                padding: 15px;
                background-color: #2a2a2a;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .collapsible-header {
                display: flex;
                align-items: center;
                padding: 10px;
                background-color: #333;
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            
            .collapsible-header:hover {
                background-color: #444;
            }
            
            .collapsible-content {
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 4px;
                margin-bottom: 15px;
                margin-left: 15px;
                border-left: 2px solid #61dafb;
            }
            
            .command-element {
                background-color: #2a2a2a;
                padding: 8px 12px;
                border-radius: 4px;
                margin: 5px 0;
                font-family: 'Consolas', 'Courier New', monospace;
                border-left: 3px solid #61dafb;
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
                background-color: #61dafb;
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
                background-color: #28a745;
                color: #fff;
            }
            
            .status-tag.in-progress {
                background-color: #ffc107;
                color: #000;
            }
            
            .status-tag.error {
                background-color: #dc3545;
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
                color: #fff;
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
                background-color: #333;
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
                color: #61dafb;
            }
            
            .brand-slogan {
                font-size: 1rem;
                color: #888;
                margin-bottom: 30px;
                text-align: center;
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

# Terminal view component
def create_terminal_view(content):
    return html.Div(
        className="terminal-view",
        children=[
            html.Pre(content)
        ]
    )

# Editor view component
def create_editor_view(filename, content, language="text"):
    syntax_highlighting = {
        "python": {
            "keywords": ["def", "class", "import", "from", "return", "if", "else", "elif", "for", "while", "try", "except", "with"],
            "keyword_color": "#569CD6",
            "string_color": "#CE9178",
            "comment_color": "#6A9955",
            "function_color": "#DCDCAA",
            "variable_color": "#9CDCFE"
        },
        "json": {
            "keywords": ["null", "true", "false"],
            "keyword_color": "#569CD6",
            "string_color": "#CE9178",
            "number_color": "#B5CEA8",
            "punctuation_color": "#D4D4D4"
        },
        "text": {
            "color": "#D4D4D4"
        }
    }
    
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

# Collapsible section component
def create_collapsible_section(id_prefix, header_content, content, is_open=True):
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

# Command execution component
def create_command_element(command, status="completed"):
    icon_class = "fas fa-check-circle text-success" if status == "completed" else "fas fa-spinner fa-spin text-warning"
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span("Executing command", style={"marginRight": "10px"}),
            html.Code(command, className="command-element")
        ]
    )

# File creation/editing component
def create_file_operation(operation, filepath, status="completed"):
    icon_class = "fas fa-check-circle text-success" if status == "completed" else "fas fa-spinner fa-spin text-warning"
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span(f"{operation} file", style={"marginRight": "10px"}),
            html.Code(filepath, className="file-path")
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
                # Header with current task (this will be dynamically replaced)
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
                        html.H3(id="project-title", children="", style={"margin": "0 0 10px 0"})
                    ]
                ),
                
                # Initial thinking indicator - will be replaced by agent-generated content
                html.Div(
                    className="chat-message",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "color": "#888"
                            },
                            children=[
                                html.I(className="fas fa-circle-notch fa-spin", style={"marginRight": "10px"}),
                                html.Span("Processing your request...")
                            ]
                        )
                    ]
                )
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
                            className="btn-control"
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
                        html.Span("Terminal"),
                        html.Div(
                            style={
                                "marginLeft": "20px",
                                "display": "flex",
                                "alignItems": "center",
                                "color": "#888",
                                "fontSize": "0.85em"
                            },
                            children=[
                                html.Span("Initializing"),
                                html.Code(
                                    "workspace",
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
                    children=[
                        # Default to terminal view with initialization message
                        create_terminal_view("Initializing AgenDev system...\nWaiting for your input...")
                    ]
                ),
                
                # Controls
                html.Div(
                    className="view-controls",
                    children=[
                        html.Div(
                            className="progress-controls",
                            children=[
                                html.Button(
                                    html.I(className="fas fa-step-backward"),
                                    className="btn-control",
                                    id="playback-backward"
                                ),
                                html.Button(
                                    html.I(className="fas fa-play"),
                                    className="btn-control",
                                    id="playback-play"
                                ),
                                html.Button(
                                    html.I(className="fas fa-step-forward"),
                                    className="btn-control",
                                    id="playback-forward"
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="playback-slider",
                                        min=0,
                                        max=100,
                                        value=0,
                                        updatemode="drag",
                                        marks=None,
                                        tooltip={"always_visible": False},
                                        className="timeline-slider"
                                    ),
                                    style={"width": "300px", "marginLeft": "10px", "marginRight": "10px"}
                                )
                            ]
                        ),
                        html.Div(
                            id="status-indicator",
                            className="status-indicator",
                            children=[
                                html.Span(
                                    html.I(className="fas fa-circle-notch fa-spin"),
                                    className="status-tag in-progress",
                                    style={"marginRight": "5px"}
                                ),
                                html.Span("Waiting for input...")
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Store of playback data with timeline steps
playback_store = dcc.Store(
    id='playback-data',
    data={"total_steps": 0, "current_step": 0, "is_playing": False, "steps": []}
)

# Add app state store
app_state_store = dcc.Store(
    id='app-state',
    data={"view": "landing", "initial_prompt": "", "project_id": None, "tasks": []}
)

# Add a Store for action recording
action_record_store = dcc.Store(
    id='action-record',
    data={"actions": [], "current_index": -1, "is_recording": False, "is_playing": False}
)

# Set the app layout - this is what was missing
app.layout = html.Div([
    app_state_store,
    playback_store,
    action_record_store,
    landing_page,
    main_view
])

# Utility function to create task elements based on a plan
def create_task_elements(plan, status="in-progress"):
    """Create UI elements for task visualization based on the implementation plan."""
    task_elements = []
    
    if not plan or "phases" not in plan:
        return [html.P("No plan available")]
    
    for phase_idx, phase in enumerate(plan.get("phases", [])):
        phase_elements = []
        
        # Add phase description
        phase_elements.append(html.P(phase.get("description", "")))
        
        # Add steps for this phase
        for step_idx, step in enumerate(phase.get("steps", [])):
            step_status = "completed" if phase_idx == 0 and step_idx < len(phase.get("steps", [])) // 2 else status
            
            # Add commands if available
            for cmd in step.get("commands", []):
                phase_elements.append(create_command_element(cmd, status=step_status))
            
            # Add file operations if needed (just a placeholder example)
            if "files" in step:
                for file_info in step.get("files", []):
                    file_path = file_info.get("path", "")
                    operation = file_info.get("operation", "Creating")
                    phase_elements.append(create_file_operation(operation, file_path, status=step_status))
            
            # Add a summary of the step
            phase_elements.append(html.P(step.get("description", "")))
        
        task_elements.extend(phase_elements)
    
    return task_elements

# Function to record an action in the timeline
def record_action(action_type, content, metadata=None):
    """Record an action for the playback system.
    
    Args:
        action_type (str): Type of action (e.g., 'terminal', 'editor', 'file_operation')
        content (str): The content to display (terminal output, file content, etc.)
        metadata (dict, optional): Additional information about the action.
    
    Returns:
        dict: The action record
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    action = {
        "id": str(uuid.uuid4()),
        "type": action_type,
        "content": content,
        "metadata": metadata or {},
        "timestamp": timestamp
    }
    
    return action

# Enhanced process_user_prompt to use all agents properly
async def process_user_prompt(prompt):
    """Process the user prompt through our complete agent system and return results with recorded actions."""
    actions = []
    try:
        # Generate a project ID
        project_id = str(uuid.uuid4())
        workspace_dir = os.path.join(workspace_path, f"project_{project_id[:8]}")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Record initial action
        actions.append(record_action(
            "system", 
            f"Received user prompt: {prompt}", 
            {"status": "started"}
        ))
        
        # Step 1: Analyze requirements with the PlannerAgent
        actions.append(record_action(
            "system", 
            "Analyzing requirements...", 
            {"status": "in-progress", "agent": "planner"}
        ))
        
        analysis_result = await planner_agent.process({
            "action": "analyze_requirements",
            "prompt": prompt,
            "workspace": workspace_dir
        })
        
        if not analysis_result.get("success", False):
            actions.append(record_action(
                "system", 
                f"Failed to analyze requirements: {analysis_result.get('error', 'Unknown error')}", 
                {"status": "error", "agent": "planner"}
            ))
            return {
                "success": False,
                "error": analysis_result.get("error", "Failed to analyze requirements"),
                "project_id": project_id,
                "actions": actions
            }
        
        # Record successful requirements analysis
        requirements_content = json.dumps(analysis_result.get("requirements", []), indent=2)
        actions.append(record_action(
            "terminal", 
            f"✅ Requirements Analysis Complete\n\n{requirements_content}", 
            {"status": "completed", "agent": "planner"}
        ))
        
        # Step 2: Generate an implementation plan
        actions.append(record_action(
            "system", 
            "Creating implementation plan...", 
            {"status": "in-progress", "agent": "planner"}
        ))
        
        plan_result = await planner_agent.process({
            "action": "create_plan",
            "requirements": analysis_result.get("requirements", []),
            "workspace": workspace_dir
        })
        
        if not plan_result.get("success", False):
            actions.append(record_action(
                "system", 
                f"Failed to create implementation plan: {plan_result.get('error', 'Unknown error')}", 
                {"status": "error", "agent": "planner"}
            ))
            return {
                "success": False,
                "error": plan_result.get("error", "Failed to create implementation plan"),
                "project_id": project_id,
                "actions": actions
            }
        
        # Record successful plan creation
        plan_content = json.dumps(plan_result.get("plan", {}), indent=2)
        actions.append(record_action(
            "editor", 
            f"# Implementation Plan\n\n```json\n{plan_content}\n```", 
            {"status": "completed", "agent": "planner", "filename": "implementation_plan.md"}
        ))
        
        # Store project information
        project_info = {
            "project_id": project_id,
            "project_name": plan_result.get("plan", {}).get("project_name", "New Project"),
            "description": prompt,
            "requirements": analysis_result.get("requirements", []),
            "technologies": analysis_result.get("technologies", []),
            "plan": plan_result.get("plan", {}),
            "workspace": workspace_dir,
            "tasks": []
        }
        
        # Create task breakdown and execute tasks
        tasks = []
        phases = project_info["plan"].get("phases", [])
        
        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get("name", f"Phase {phase_idx+1}")
            phase_desc = phase.get("description", "")
            
            # Record phase processing
            actions.append(record_action(
                "system", 
                f"Processing phase: {phase_name}", 
                {"status": "in-progress", "phase": phase_idx}
            ))
            
            # Define task status based on position
            task_status = "completed" if phase_idx == 0 else "in-progress" if phase_idx == 1 else "pending"
            
            # Create task object
            task = {
                "id": f"task{phase_idx+1}",
                "name": phase_name,
                "description": phase_desc,
                "status": task_status,
                "elements": create_task_elements({"phases": [phase]}, status=task_status)
            }
            tasks.append(task)
            
            # Execute completed and in-progress tasks
            if task_status in ["completed", "in-progress"]:
                # Execute task according to phase type
                if "setup" in phase_name.lower() or "environment" in phase_name.lower():
                    # Environment setup with CodeAgent
                    actions.append(record_action(
                        "system", 
                        f"Setting up environment for: {phase_name}", 
                        {"status": "in-progress", "agent": "code", "phase": phase_idx}
                    ))
                    
                    # Execute using CodeAgent
                    setup_result = await code_agent.process({
                        "action": "initialize_context",
                        "project_name": project_info["project_name"],
                        "technologies": project_info.get("technologies", []),
                        "workspace_path": workspace_dir
                    })
                    
                    if setup_result.get("success", False):
                        # Record setup commands
                        setup_commands = setup_result.get("commands", [])
                        if setup_commands:
                            cmd_output = "\n".join([f"$ {cmd}" for cmd in setup_commands])
                            actions.append(record_action(
                                "terminal", 
                                f"Setting up environment:\n{cmd_output}", 
                                {"status": "completed", "agent": "code", "phase": phase_idx}
                            ))
                
                elif "implementation" in phase_name.lower() or "code" in phase_name.lower():
                    # Implementation with CodeAgent
                    actions.append(record_action(
                        "system", 
                        f"Implementing code for: {phase_name}", 
                        {"status": "in-progress", "agent": "code", "phase": phase_idx}
                    ))
                    
                    # Execute implementation for each step in the phase
                    for step_idx, step in enumerate(phase.get("steps", [])):
                        step_name = step.get("name", f"Step {step_idx+1}")
                        
                        # Only process a subset of steps for in-progress tasks
                        if task_status == "in-progress" and step_idx > len(phase.get("steps", [])) // 2:
                            continue
                        
                        # Use CodeAgent to implement this step
                        implement_result = await code_agent.process({
                            "action": "implement_task",
                            "task_name": step_name,
                            "task_description": step.get("description", ""),
                            "workspace_path": workspace_dir
                        })
                        
                        if implement_result.get("success", False):
                            # Record file operations from the implementation
                            for file_op in implement_result.get("file_operations", []):
                                file_path = file_op.get("file_path", "")
                                content = file_op.get("content", "")
                                operation = file_op.get("operation", "create")
                                
                                if file_path and content:
                                    actions.append(record_action(
                                        "editor", 
                                        content, 
                                        {
                                            "status": "completed", 
                                            "agent": "code", 
                                            "filename": os.path.basename(file_path),
                                            "filepath": file_path,
                                            "operation": operation,
                                            "phase": phase_idx
                                        }
                                    ))
                
                elif "integration" in phase_name.lower():
                    # Integration with IntegrationAgent
                    actions.append(record_action(
                        "system", 
                        f"Integrating components for: {phase_name}", 
                        {"status": "in-progress", "agent": "integration", "phase": phase_idx}
                    ))
                    
                    # Use IntegrationAgent to handle integration
                    integration_result = await integration_agent.process({
                        "action": "integrate_task",
                        "project_name": project_info["project_name"],
                        "workspace_path": workspace_dir
                    })
                    
                    if integration_result.get("success", False):
                        # Record integration points
                        for point in integration_result.get("integration_points", []):
                            source = point.get("source_file", "")
                            target = point.get("target_file", "")
                            description = point.get("description", "")
                            
                            actions.append(record_action(
                                "editor", 
                                f"# Integration: {source} → {target}\n\n{description}", 
                                {
                                    "status": "completed",
                                    "agent": "integration", 
                                    "filename": f"integration_{os.path.basename(source)}_{os.path.basename(target)}.md",
                                    "phase": phase_idx
                                }
                            ))
                
                elif "test" in phase_name.lower():
                    # Testing with CodeAgent
                    actions.append(record_action(
                        "system", 
                        f"Running tests for: {phase_name}", 
                        {"status": "in-progress", "agent": "code", "phase": phase_idx}
                    ))
                    
                    # Use CodeAgent to run tests
                    test_result = await code_agent.process({
                        "action": "run_tests",
                        "workspace_path": workspace_dir
                    })
                    
                    if test_result.get("success", False):
                        # Record test results
                        test_output = test_result.get("output", "No test output available")
                        actions.append(record_action(
                            "terminal", 
                            f"Test Results:\n{test_output}", 
                            {"status": "completed", "agent": "code", "phase": phase_idx}
                        ))
                
                elif "deploy" in phase_name.lower():
                    # Deployment with DeploymentAgent
                    actions.append(record_action(
                        "system", 
                        f"Deploying project: {phase_name}", 
                        {"status": "in-progress", "agent": "deployment", "phase": phase_idx}
                    ))
                    
                    # Use DeploymentAgent to handle deployment
                    deploy_result = await deployment_agent.process({
                        "action": "deploy_project",
                        "project_name": project_info["project_name"],
                        "workspace_path": workspace_dir,
                        "platform": "local"  # Default to local for now
                    })
                    
                    if deploy_result.get("success", False):
                        # Record deployment results
                        deploy_output = deploy_result.get("output", "Deployment completed")
                        actions.append(record_action(
                            "terminal", 
                            f"Deployment Output:\n{deploy_output}", 
                            {"status": "completed", "agent": "deployment", "phase": phase_idx}
                        ))
                
                # Mark as completed if we're processing a completed task
                if task_status == "completed":
                    actions.append(record_action(
                        "system", 
                        f"Completed: {phase_name}", 
                        {"status": "completed", "phase": phase_idx}
                    ))
        
        # Store tasks in project info
        project_info["tasks"] = tasks
        active_projects[project_id] = project_info
        
        # Add final summary action
        actions.append(record_action(
            "system", 
            f"Project setup complete: {project_info['project_name']}", 
            {"status": "completed"}
        ))
        
        return {
            "success": True,
            "project_id": project_id,
            "project_name": project_info["project_name"],
            "tasks": tasks,
            "actions": actions
        }
    
    except Exception as e:
        error_message = f"Error processing prompt: {e}\n{traceback.format_exc()}"
        print(error_message)
        
        # Record error action
        actions.append(record_action(
            "system", 
            error_message, 
            {"status": "error"}
        ))
        
        return {
            "success": False,
            "error": str(e),
            "project_id": None,
            "actions": actions
        }

# Updated transition_to_main_view callback for better error handling
@app.long_callback(
    [Output("app-state", "data", allow_duplicate=True),
     Output("landing-page", "style", allow_duplicate=True),
     Output("main-container", "style", allow_duplicate=True),
     Output("project-title", "children", allow_duplicate=True),
     Output("chat-container", "children", allow_duplicate=True),
     Output("action-record", "data", allow_duplicate=True)],
    [Input("submit-button", "n_clicks")],
    [State("initial-prompt", "value"),
     State("app-state", "data")],
    prevent_initial_call=True,
    running=[
        (Output("submit-button", "disabled"), True, False),
        (Output("submit-button", "children"), "Processing...", "Submit")
    ],
    # Add manager-specific config to help with process handling
    manager_config={
        "processes": 1,
        "max_processes": 2,
        "disable_job_param": True,
        "job_timeout": 1800  # 30 minutes should be plenty
    }
)
def transition_to_main_view(n_clicks, prompt_value, current_state):
    if not n_clicks:
        raise PreventUpdate
    
    if not prompt_value:
        raise PreventUpdate
    
    try:
        print(f"Processing prompt: {prompt_value}")
        
        # Initialize or get the current state
        if not current_state:
            current_state = {"projects": {}}
            
        # Generate a unique project ID
        project_id = str(uuid.uuid4())
        
        # Set up the initial state for the project
        current_state["current_project"] = project_id
        current_state["projects"][project_id] = {
            "id": project_id,
            "title": prompt_value,
            "prompt": prompt_value,
            "actions": [],
            "status": "in-progress",
            "timestamp": datetime.now().isoformat()
        }
        
        # Create the workspace directory for this project
        workspace_dir = os.path.join(workspace_path, f"project_{project_id[:8]}")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Set up the initial action record
        action_record = {
            "actions": [],
            "current_index": 0,
            "is_playing": False,
            "auto_advance": False
        }
        
        # Update state
        current_state["view"] = "main"
        current_state["initial_prompt"] = prompt_value
        
        # Update the app state with project info
        current_state["project_id"] = project_id
        
        # Hide landing page, show main container
        landing_style = {"display": "none"}
        main_style = {"display": "flex"}
        
        # We'll use "Processing..." as the initial project title
        project_name = "Processing your request..."
        
        # Create initial chat container children with system message and "thinking" indicator
        chat_children = [
            # System message with initial title
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
                    html.H3(id="project-title", children=project_name, style={"margin": "0 0 10px 0"})
                ]
            ),
            # Add thinking indicator
            html.Div(
                className="chat-message",
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "color": "#888"
                        },
                        children=[
                            html.I(className="fas fa-circle-notch fa-spin", style={"marginRight": "10px"}),
                            html.Span("Processing your request... This may take a few minutes.")
                        ]
                    )
                ]
            )
        ]
        
        # Start processing in background - don't use the long callback's process
        # as it can cause issues with the diskcache manager
        background_process_prompt(prompt_value, project_id, workspace_dir)
        
        return current_state, landing_style, main_style, project_name, chat_children, action_record
    
    except Exception as e:
        error_message = f"Error in transition: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        
        # Provide a fallback return in case of errors
        current_state["view"] = "main"
        current_state["initial_prompt"] = prompt_value
        landing_style = {"display": "none"}
        main_style = {"display": "flex"}
        project_name = "Error Processing Request"
        
        # Show error in UI
        error_children = [
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
                    html.H3(id="project-title", children=project_name, style={"margin": "0 0 10px 0"})
                ]
            ),
            html.Div(
                className="chat-message",
                style={"backgroundColor": "#dc3545", "color": "white", "padding": "15px", "borderRadius": "8px"},
                children=[
                    html.P("An error occurred while processing your request:"),
                    html.Pre(str(e), style={"maxHeight": "200px", "overflow": "auto"})
                ]
            )
        ]
        
        error_action_data = {
            "actions": [record_action("system", f"Error: {str(e)}", {"status": "error"})],
            "current_index": 0,
            "is_recording": False,
            "is_playing": False,
            "playback_speed": 1.0
        }
        
        return current_state, landing_style, main_style, project_name, error_children, error_action_data

# Replace the background_process_prompt function with a more robust implementation
def background_process_prompt(prompt, project_id, workspace_dir):
    """Process the prompt in the background and update app state"""
    try:
        print(f"Starting background processing for project {project_id}")
        
        def run_async_task():
            try:
                print(f"Background thread started for project {project_id}")
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Process the prompt
                try:
                    print(f"Running process_user_prompt for {project_id}")
                    result = loop.run_until_complete(process_user_prompt(prompt))
                    print(f"process_user_prompt completed for {project_id}")
                    
                    # Update the app state with the results - safely
                    if project_id in active_projects:
                        print(f"Project {project_id} already in active_projects")
                        # Update actions if they exist in the result
                        if "actions" in result and isinstance(result["actions"], list):
                            print(f"Updating actions for project {project_id}: {len(result['actions'])} actions")
                            active_projects[project_id]["actions"] = result["actions"]
                    else:
                        # Store the project info
                        print(f"Creating new project entry for {project_id}")
                        active_projects[project_id] = {
                            "project_id": project_id,
                            "project_name": result.get("project_name", "New Project"),
                            "description": prompt,
                            "workspace": workspace_dir,
                            "actions": result.get("actions", [])
                        }
                    print(f"active_projects now contains: {list(active_projects.keys())}")
                    
                except Exception as e:
                    error_message = f"Error in process_user_prompt: {e}\n{traceback.format_exc()}"
                    print(error_message)
                    
                    # Record error
                    error_action = record_action(
                        "system", 
                        error_message, 
                        {"status": "error"}
                    )
                    
                    # Store error in projects
                    if project_id in active_projects:
                        if "actions" in active_projects[project_id]:
                            active_projects[project_id]["actions"].append(error_action)
                        else:
                            active_projects[project_id]["actions"] = [error_action]
                    else:
                        active_projects[project_id] = {
                            "project_id": project_id,
                            "project_name": "Error",
                            "description": prompt,
                            "workspace": workspace_dir,
                            "actions": [error_action]
                        }
                finally:
                    # Always clean up the event loop
                    try:
                        loop.close()
                        print(f"Event loop closed for project {project_id}")
                    except Exception as e:
                        print(f"Error closing event loop: {e}")
                        pass  # Ensure this doesn't raise
            except Exception as e:
                error_message = f"Error in background thread: {e}\n{traceback.format_exc()}"
                print(error_message)
                
                # Record error
                error_action = record_action(
                    "system", 
                    error_message, 
                    {"status": "error"}
                )
                
                # Store error in projects
                if project_id in active_projects:
                    if "actions" in active_projects[project_id]:
                        active_projects[project_id]["actions"].append(error_action)
                    else:
                        active_projects[project_id]["actions"] = [error_action]
                else:
                    active_projects[project_id] = {
                        "project_id": project_id,
                        "project_name": "Error",
                        "description": prompt,
                        "workspace": workspace_dir,
                        "actions": [error_action]
                    }
        
        # Use ThreadPoolExecutor instead of raw thread creation
        future = background_executor.submit(run_async_task)
        # Don't wait for the result - it's a background task
        print(f"Background task submitted for project {project_id}")
        
    except Exception as e:
        error_message = f"Failed to start background processing: {e}\n{traceback.format_exc()}"
        print(error_message)
        
        # Record error
        error_action = record_action(
            "system", 
            error_message, 
            {"status": "error"}
        )
        
        # Store error in projects
        if project_id in active_projects:
            if "actions" in active_projects[project_id]:
                active_projects[project_id]["actions"].append(error_action)
            else:
                active_projects[project_id]["actions"] = [error_action]
        else:
            active_projects[project_id] = {
                "project_id": project_id,
                "project_name": "Error",
                "description": prompt,
                "workspace": workspace_dir,
                "actions": [error_action]
            }

# Restore the playback controls callback
@app.callback(
    [Output("view-content", "children", allow_duplicate=True),
     Output("view-type-indicator", "children", allow_duplicate=True),
     Output("status-indicator", "children", allow_duplicate=True),
     Output("playback-slider", "value", allow_duplicate=True),
     Output("action-record", "data", allow_duplicate=True)],
    [Input("playback-backward", "n_clicks"),
     Input("playback-play", "n_clicks"),
     Input("playback-forward", "n_clicks"),
     Input("playback-slider", "value")],
    [State("action-record", "data"),
     State("app-state", "data")],
    prevent_initial_call=True
)
def handle_playback_controls(backward_clicks, play_clicks, forward_clicks, slider_value, action_record, app_state):
    """Handle playback control interactions."""
    triggered_id = ctx.triggered_id if ctx.triggered_id is not None else 'No clicks yet'
    
    # Safety check for empty action_record
    if not action_record:
        action_record = {"actions": [], "current_index": 0, "is_playing": False}
    
    # Make a copy of the action record to modify
    action_record = action_record.copy()
    actions = action_record.get("actions", [])
    current_index = action_record.get("current_index", 0)
    is_playing = action_record.get("is_playing", False)
    
    if not actions:
        # No actions to play back
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Calculate max slider value
    max_index = len(actions) - 1
    if max_index < 0:
        max_index = 0
    
    # Handle different control clicks
    if triggered_id == "playback-backward" and current_index > 0:
        # Step backward
        current_index -= 1
        is_playing = False
    elif triggered_id == "playback-forward" and current_index < max_index:
        # Step forward
        current_index += 1
        is_playing = False
    elif triggered_id == "playback-play":
        # Toggle play/pause
        is_playing = not is_playing
    elif triggered_id == "playback-slider":
        # Slider moved directly
        # Convert slider percentage to index
        if max_index > 0:
            current_index = round(slider_value / 100 * max_index)
        else:
            current_index = 0
        is_playing = False
    
    # Update action record
    action_record["current_index"] = current_index
    action_record["is_playing"] = is_playing
    
    # Get current action
    current_action = actions[current_index] if actions and 0 <= current_index < len(actions) else None
    
    if not current_action:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, action_record
    
    # Calculate slider value (percentage)
    slider_percentage = (current_index / max_index * 100) if max_index > 0 else 0
    
    # Prepare view content based on action type
    action_type = current_action.get("type", "system")
    content = current_action.get("content", "")
    metadata = current_action.get("metadata", {})
    
    if action_type == "terminal":
        view_content = create_terminal_view(content)
        view_type = [
            html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
            html.Span("Terminal"),
            html.Div(
                style={
                    "marginLeft": "20px",
                    "display": "flex",
                    "alignItems": "center",
                    "color": "#888",
                    "fontSize": "0.85em"
                },
                children=[
                    html.Span("Executing commands"),
                    html.Code(
                        metadata.get("agent", "system"),
                        style={
                            "marginLeft": "5px",
                            "backgroundColor": "transparent",
                            "padding": "0"
                        }
                    )
                ]
            )
        ]
    elif action_type == "editor":
        filename = metadata.get("filename", "file.txt")
        view_content = create_editor_view(
            filename,
            content,
            "python" if filename.endswith(".py") else "json" if filename.endswith(".json") else "text"
        )
        
        operation = metadata.get("operation", "creating")
        if operation == "modify":
            operation_text = "Editing"
        elif operation == "delete":
            operation_text = "Deleting"
        else:
            operation_text = "Creating"
        
        view_type = [
            html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
            html.Span("Editor"),
            html.Div(
                style={
                    "marginLeft": "20px",
                    "display": "flex",
                    "alignItems": "center",
                    "color": "#888",
                    "fontSize": "0.85em"
                },
                children=[
                    html.Span(f"{operation_text} file"),
                    html.Code(
                        filename,
                        style={
                            "marginLeft": "5px",
                            "backgroundColor": "transparent",
                            "padding": "0"
                        }
                    )
                ]
            )
        ]
    else:  # system or other types
        # For system messages, show clear indication of which agent is working
        agent = metadata.get("agent", "system")
        agent_message = f"[{agent.upper()}] " if agent != "system" else ""
        
        view_content = create_terminal_view(f"{agent_message}{content}")
        view_type = [
            html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
            html.Span("System"),
            html.Div(
                style={
                    "marginLeft": "20px",
                    "display": "flex",
                    "alignItems": "center",
                    "color": "#888",
                    "fontSize": "0.85em"
                },
                children=[
                    html.Span("Processing"),
                    html.Code(
                        agent,
                        style={
                            "marginLeft": "5px",
                            "backgroundColor": "transparent",
                            "padding": "0"
                        }
                    )
                ]
            )
        ]
    
    # Update status indicator
    status = metadata.get("status", "pending")
    phase = metadata.get("phase", None)
    
    # Build status text with phase info if available
    status_text = f"Action {current_index+1}/{len(actions)}"
    if phase is not None:
        status_text += f" (Phase {phase+1})"
    
    if status == "completed":
        status_indicator = [
            html.Span(
                html.I(className="fas fa-check-circle"),
                className="status-tag success",
                style={"marginRight": "5px"}
            ),
            html.Span(status_text),
            html.Span(current_action.get("timestamp", ""), className="time-indicator")
        ]
    elif status == "in-progress":
        status_indicator = [
            html.Span(
                html.I(className="fas fa-spinner fa-spin"),
                className="status-tag in-progress",
                style={"marginRight": "5px"}
            ),
            html.Span(status_text),
            html.Span(current_action.get("timestamp", ""), className="time-indicator")
        ]
    elif status == "error":
        status_indicator = [
            html.Span(
                html.I(className="fas fa-times-circle"),
                className="status-tag error",
                style={"marginRight": "5px"}
            ),
            html.Span(status_text),
            html.Span(current_action.get("timestamp", ""), className="time-indicator")
        ]
    else:
        status_indicator = [
            html.Span(
                html.I(className="fas fa-circle"),
                className="status-tag",
                style={"marginRight": "5px"}
            ),
            html.Span(status_text),
            html.Span(current_action.get("timestamp", ""), className="time-indicator")
        ]
    
    return view_content, view_type, status_indicator, slider_percentage, action_record

# Auto playback interval 
@app.callback(
    Output("action-record", "data", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    State("action-record", "data"),
    prevent_initial_call=True
)
def update_playback_automatically(n_intervals, action_record):
    """Automatically advance playback when in play mode."""
    if not action_record or not action_record.get("is_playing", False):
        return dash.no_update
    
    actions = action_record.get("actions", [])
    current_index = action_record.get("current_index", 0)
    max_index = len(actions) - 1
    
    if current_index < max_index:
        action_record["current_index"] = current_index + 1
        return action_record
    else:
        # Stop playing when we reach the end
        action_record["is_playing"] = False
        return action_record

# Add an interval component for auto playback
interval = dcc.Interval(
    id='interval-component',
    interval=1000,  # 1 second
    n_intervals=0,
    disabled=True
)

# Add a new interval component for UI updates
interval_ui = dcc.Interval(
    id='update-interval',
    interval=1000,  # 1 second
    n_intervals=0,
    disabled=True
)

# Update the app layout to include the new interval
app.layout = html.Div([
    app_state_store,
    playback_store,
    action_record_store,
    interval_ui,
    interval,
    landing_page,
    main_view
])

# Restore the task collapsible section callbacks
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

# Callback for task2 collapsible section  
@app.callback(
    Output("task2-content", "style"),
    Input("task2-header", "n_clicks"),
    State("task2-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task2(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Callback for task3 collapsible section
@app.callback(
    Output("task3-content", "style"),
    Input("task3-header", "n_clicks"),
    State("task3-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task3(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Update the run_server section to forcefully disable all hot reloading
if __name__ == "__main__":
    # Make sure to install required packages
    try:
        import nest_asyncio
    except ImportError:
        import pip
        pip.main(['install', 'nest_asyncio'])
        import nest_asyncio
    
    # Apply nest_asyncio only once at startup
    nest_asyncio.apply()
    
    # Set threading and process safety 
    import multiprocessing
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # Method already set
            pass
    
    # Set environment variables to disable hot reloading
    import os
    os.environ['DASH_HOT_RELOAD'] = 'false'
    os.environ['FLASK_ENV'] = 'production'
    
    # Completely disable Flask's dev server auto-reloading
    import flask
    flask.Flask.debug = False
    
    # Use Dash in "production" mode rather than development mode
    # This is the most effective way to prevent auto-reloading
    
    # Run with ALL auto-refresh features disabled
    app.run_server(
        debug=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False,
        dev_tools_serve_dev_bundles=False,
        use_reloader=False,
        host='0.0.0.0'  # Listen on all network interfaces
    )