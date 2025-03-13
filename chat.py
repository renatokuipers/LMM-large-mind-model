import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime
import os
import sys
from pathlib import Path

# Ensure we can import AgenDev modules
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import AgenDev core components
from src.agendev.core import AgenDev, AgenDevConfig, ProjectState
from src.agendev.models.task_models import Task, TaskStatus, TaskPriority, TaskRisk, TaskType, Epic, TaskGraph
from src.agendev.models.planning_models import SimulationConfig, PlanSnapshot
from src.agendev.snapshot_engine import SnapshotEngine
from src.agendev.parameter_controller import ParameterController
from src.agendev.utils.fs_utils import resolve_path

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
                        html.H3(id="project-title", children="Python Snake Game Development", style={"margin": "0 0 10px 0"})
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task1",
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"marginRight": "10px", "color": "#28a745"}),
                        html.Span("Continue: Create Next.js application for political debate simulation")
                    ]),
                    [
                        html.P("Creating a Next.js application for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_command_element("cd /home/ubuntu && create_nextjs_app zelenskyy_debate_sim"),
                        html.P("Successfully created Next.js application for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/app/data/scenarios.json")
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task2",
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"marginRight": "10px", "color": "#28a745"}),
                        html.Span("Develop dialogue system and political scenario scripts")
                    ]),
                    [
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Editing", "zelenskyy_debate_sim/src/app/page.tsx"),
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/app/simulation/page.tsx"),
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Editing", "zelenskyy_debate_sim/src/app/globals.css")
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task3",
                    html.Div([
                        html.I(className="fas fa-spinner fa-spin", style={"marginRight": "10px", "color": "#ffc107"}),
                        html.Span("Design and implement user interface with styling")
                    ]),
                    [
                        html.P("Moving to add additional user interface elements and styling to the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/CharacterPortrait.tsx"),
                        html.P("Adding user interface components and styling to enhance the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/DialogueBubble.tsx"),
                        html.P("Adding user interface components and styling to enhance the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/ResponseOption.tsx")
                    ]
                ),
                
                # Thinking indicator
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
                                html.Span("Thinking")
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
                    style={
                        "padding": "5px 10px",
                        "backgroundColor": "#2d2d2d",
                        "borderBottom": "1px solid #444",
                        "display": "flex",
                        "alignItems": "center"
                    },
                    children=[
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
                                html.Span("Creating file"),
                                html.Code(
                                    "zelenskyy_debate_sim/src/app/data/scenarios.json",
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
                        # Default to editor view
                        create_editor_view(
                            "scenarios.json",
                            '''
{
  "scenarios": [
    {
      "id": 1,
      "title": "Opening Remarks",
      "description": "President Trump welcomes you to the White House. The meeting has just begun with initial pleasantries.",
      "trumpDialogue": "President Trump welcomes you to the White House. We're going to have a great discussion today about ending this terrible war. I hope I'm going to be remembered as a peacemaker.",
      "vanceDialogue": "",
      "options": [
        {
          "id": "1a",
          "text": "Thank you, Mr. President. Ukraine is grateful for America's support. We look forward to discussing how we can achieve a just peace that ensures Ukraine's security.",
          "type": "diplomatic",
          "trumpReaction": "positive",
          "vanceReaction": "neutral",
          "nextScenario": 2
        },
        {
          "id": "1b",
          "text": "Thank you for meeting with me. I must emphasize that Ukraine needs more than just words - we need continued military support and security guarantees to end this war.",
          "type": "assertive",
          "trumpReaction": "neutral",
          "vanceReaction": "negative",
          "nextScenario": 2
        },
        ...
      ]
    }
  ]
}''',
                            "json"
                        )
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
                                        value=50,
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
                            className="status-indicator",
                            children=[
                                html.Span(
                                    html.I(className="fas fa-check-circle"),
                                    className="status-tag success",
                                    style={"marginRight": "5px"}
                                ),
                                html.Span("Deploy simulation to a public URL for permanent access"),
                                html.Span("9/9", className="time-indicator")
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Initialize AgenDev system
class AgenDevUI:
    """Integration between Dash UI and AgenDev core system"""
    
    def __init__(self):
        """Initialize the AgenDev UI integration"""
        self.agendev_instance = None
        self.config = None
        self.snapshot_engine = None
        self.current_tasks = []
        self.action_history = []
        self.current_view = "terminal"  # Default view
        
    def initialize_project(self, project_name, project_description):
        """Initialize a new AgenDev project"""
        # Create a workspace directory based on project name
        workspace_dir = os.path.join("workspace", project_name.lower().replace(' ', '_'))
        
        # Initialize AgenDev configuration
        self.config = AgenDevConfig(
            project_name=project_name,
            workspace_dir=workspace_dir
        )
        
        # Initialize AgenDev instance
        try:
            self.agendev_instance = AgenDev(self.config)
            self.snapshot_engine = SnapshotEngine(workspace_dir=workspace_dir)
            
            # Record this action
            self.record_action(
                action_type="initialization",
                content=f"Initialized AgenDev project: {project_name}\n\nDescription: {project_description}",
                view_type="terminal"
            )
            
            # Create initial tasks based on project description
            self._create_initial_tasks(project_description)
            
            return {
                "success": True,
                "project_name": project_name,
                "task_count": len(self.current_tasks)
            }
        except Exception as e:
            print(f"Error initializing AgenDev project: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_initial_tasks(self, project_description):
        """Create initial tasks based on project description"""
        try:
            # Define core tasks for the project
            tasks = [
                {
                    "title": "Initialize project directory",
                    "description": "Set up project directory and initial file structure",
                    "type": TaskType.PLANNING,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.LOW,
                    "duration": 0.5
                },
                {
                    "title": "Create basic project structure",
                    "description": "Set up directories and configuration files",
                    "type": TaskType.PLANNING,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.LOW,
                    "duration": 0.5
                },
                {
                    "title": f"Implement core functionality",
                    "description": f"Write code for main features based on: {project_description}",
                    "type": TaskType.IMPLEMENTATION,
                    "priority": TaskPriority.HIGH,
                    "risk": TaskRisk.MEDIUM,
                    "duration": 2.0
                },
                {
                    "title": "Add unit tests",
                    "description": "Write tests for the implemented functionality",
                    "type": TaskType.TEST,
                    "priority": TaskPriority.MEDIUM,
                    "risk": TaskRisk.LOW,
                    "duration": 1.0
                },
                {
                    "title": "Create documentation",
                    "description": "Write documentation for the project",
                    "type": TaskType.DOCUMENTATION,
                    "priority": TaskPriority.MEDIUM,
                    "risk": TaskRisk.LOW,
                    "duration": 1.0
                }
            ]
            
            # Add tasks to AgenDev instance
            task_ids = []
            previous_task_id = None
            
            for task in tasks:
                # Create task with optional dependency on previous task
                dependencies = [previous_task_id] if previous_task_id is not None else None
                
                task_id = self.agendev_instance.create_task(
                    title=task["title"],
                    description=task["description"],
                    task_type=task["type"],
                    priority=task["priority"],
                    risk=task["risk"],
                    estimated_duration_hours=task["duration"],
                    dependencies=dependencies
                )
                
                task_ids.append(task_id)
                previous_task_id = task_id
                
                # Add to current tasks
                self.current_tasks.append({
                    "id": str(task_id),
                    "title": task["title"],
                    "description": task["description"],
                    "status": "planned",
                    "type": task["type"].value
                })
            
            # Record this action
            self.record_action(
                action_type="task_creation",
                content=f"Created {len(tasks)} initial tasks for the project.",
                view_type="terminal"
            )
            
            # Generate implementation plan
            self.record_action(
                action_type="planning",
                content="Generating implementation plan...",
                view_type="terminal"
            )
            
            # Actually generate the plan (this might take time)
            try:
                plan = self.agendev_instance.generate_implementation_plan()
                
                # Record planning result
                self.record_action(
                    action_type="planning_complete",
                    content=f"Implementation plan generated with {len(plan.task_sequence)} tasks.\nEstimated duration: {plan.expected_duration_hours:.1f} hours\nConfidence: {plan.confidence_score:.2f}",
                    view_type="terminal"
                )
                
            except Exception as e:
                print(f"Error generating implementation plan: {e}")
                self.record_action(
                    action_type="planning_error",
                    content=f"Error generating implementation plan: {e}",
                    view_type="terminal"
                )
            
        except Exception as e:
            print(f"Error creating initial tasks: {e}")
            self.record_action(
                action_type="error",
                content=f"Error creating initial tasks: {e}",
                view_type="terminal"
            )
    
    def record_action(self, action_type, content, view_type="terminal", filename=None):
        """Record an action taken by the system for later playback"""
        timestamp = datetime.now().isoformat()
        action = {
            "timestamp": timestamp,
            "display_time": datetime.now().strftime("%H:%M:%S"),
            "type": action_type,
            "content": content,
            "view_type": view_type
        }
        
        if filename:
            action["filename"] = filename
            
        self.action_history.append(action)
        return len(self.action_history) - 1  # Return index of new action
    
    def get_action_history(self):
        """Get all recorded actions"""
        # Convert to playback format
        steps = []
        for action in self.action_history:
            step = {
                "timestamp": action["display_time"],
                "type": action["view_type"]
            }
            
            if action["view_type"] == "terminal":
                step["content"] = action["content"]
            elif action["view_type"] == "editor":
                step["content"] = action["content"]
                step["filename"] = action.get("filename", "unknown.txt")
            
            steps.append(step)
            
        return {
            "total_steps": len(steps),
            "current_step": len(steps) - 1 if steps else 0,
            "is_playing": False,
            "steps": steps
        }
        
    def implement_task(self, task_id):
        """Implement a specific task"""
        if not self.agendev_instance:
            return {"success": False, "error": "No active project"}
            
        try:
            # Record start of task implementation
            self.record_action(
                action_type="task_start",
                content=f"Starting implementation of task: {task_id}",
                view_type="terminal"
            )
            
            # Actually implement the task using AgenDev
            result = self.agendev_instance.implement_task(task_id)
            
            if result.get("success", False):
                # Record successful implementation
                file_path = result.get("file_path", "")
                implementation = result.get("implementation", "")
                
                # Record terminal action
                self.record_action(
                    action_type="task_progress",
                    content=f"Task {task_id} implementation in progress...\n$ mkdir -p $(dirname {file_path})",
                    view_type="terminal"
                )
                
                # Record editor action
                self.record_action(
                    action_type="code_creation",
                    content=implementation,
                    view_type="editor",
                    filename=file_path
                )
                
                # Record completion action
                self.record_action(
                    action_type="task_complete",
                    content=f"Task implementation completed successfully.\nOutput saved to: {file_path}",
                    view_type="terminal"
                )
                
                # Update task status in UI
                for task in self.current_tasks:
                    if task["id"] == task_id:
                        task["status"] = "completed"
                
                return {
                    "success": True, 
                    "task_id": task_id, 
                    "file_path": file_path
                }
            else:
                # Record failure
                error = result.get("error", "Unknown error during task implementation")
                self.record_action(
                    action_type="task_error",
                    content=f"Error implementing task: {error}",
                    view_type="terminal"
                )
                
                # Update task status in UI
                for task in self.current_tasks:
                    if task["id"] == task_id:
                        task["status"] = "failed"
                
                return {"success": False, "error": error}
                
        except Exception as e:
            print(f"Error in implement_task: {e}")
            error_msg = str(e)
            
            # Record the error
            self.record_action(
                action_type="error",
                content=f"Exception during task implementation: {error_msg}",
                view_type="terminal"
            )
            
            return {"success": False, "error": error_msg}
    
    def get_tasks(self):
        """Get all current tasks"""
        if not self.agendev_instance:
            return []
            
        # If we have the AgenDev instance, get fresh task data
        try:
            tasks = []
            for task_id, task in self.agendev_instance.task_graph.tasks.items():
                tasks.append({
                    "id": str(task_id),
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "type": task.task_type.value
                })
            return tasks
        except Exception as e:
            print(f"Error getting tasks from AgenDev: {e}")
            
        # Fall back to our cached task list
        return self.current_tasks
    
    def get_project_status(self):
        """Get the current status of the project"""
        if not self.agendev_instance:
            return {"success": False, "error": "No active project"}
            
        try:
            # Get status from AgenDev
            status = self.agendev_instance.get_project_status()
            return {"success": True, **status}
        except Exception as e:
            print(f"Error getting project status: {e}")
            return {
                "success": False,
                "error": f"Error getting project status: {e}"
            }

# Initialize the AgenDev UI integration
agendev_ui = AgenDevUI()

# Callback to transition from landing page to main view
@app.callback(
    [Output("app-state", "data"),
     Output("landing-page", "style"),
     Output("main-container", "style"),
     Output("project-title", "children"),
     Output("playback-data", "data")],
    [Input("submit-button", "n_clicks")],
    [State("initial-prompt", "value"),
     State("app-state", "data")],
    prevent_initial_call=True
)
def transition_to_main_view(n_clicks, prompt_value, current_state):
    if not n_clicks:
        raise PreventUpdate
    
    # Generate a title based on the prompt
    title = "New Project"
    if prompt_value:
        # Simple algorithm to extract a title
        if "create" in prompt_value.lower() and "snake" in prompt_value.lower() and "python" in prompt_value.lower():
            title = "Python Snake Game Development"
        elif "todo" in prompt_value.lower() or "task" in prompt_value.lower() or "list" in prompt_value.lower():
            title = "Todo List Application"
        elif "dashboard" in prompt_value.lower() or "data" in prompt_value.lower() or "visualization" in prompt_value.lower():
            title = "Data Visualization Dashboard"
        elif "web" in prompt_value.lower() or "site" in prompt_value.lower() or "app" in prompt_value.lower():
            title = "Web Application Development"
        elif "game" in prompt_value.lower():
            title = "Game Development Project"
        elif "api" in prompt_value.lower() or "backend" in prompt_value.lower() or "server" in prompt_value.lower():
            title = "API Development Project"
        else:
            # Extract key words for a generic title
            words = prompt_value.split()
            if len(words) > 3:
                # Take a few significant words from the middle of the prompt
                middle_index = len(words) // 2
                title_words = words[max(0, middle_index-1):min(len(words), middle_index+2)]
                title = " ".join(word.capitalize() for word in title_words) + " Project"
            else:
                # For short prompts, use the whole thing
                title = prompt_value.capitalize()
    
    # Update state
    current_state["view"] = "main"
    current_state["initial_prompt"] = prompt_value
    current_state["project_title"] = title
    
    # Initialize the AgenDev project
    try:
        result = agendev_ui.initialize_project(title, prompt_value)
        if not result.get("success", False):
            print(f"Warning: Failed to initialize AgenDev project: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error initializing AgenDev project: {e}")
    
    # Get action history for playback
    playback_data = agendev_ui.get_action_history()
    
    # Hide landing page, show main container
    landing_style = {"display": "none"}
    main_style = {"display": "flex"}
    
    return current_state, landing_style, main_style, title, playback_data

# Callback for task1 collapsible section
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

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)