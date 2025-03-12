import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
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
import threading
from datetime import datetime
import logging

# Import AgenDev components
from agendev.core import AgenDev, AgenDevConfig, ProjectState
from agendev.models.task_models import TaskType, TaskStatus, TaskPriority, TaskRisk
from agendev.tts_notification import NotificationPriority, NotificationType
from agendev.utils.fs_utils import resolve_path
from agendev.llm_module import LLMClient, Message
from agendev.tts_module import TTSClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.error(f"Error initializing AgenDev: {e}")
    # Create a minimal instance for UI rendering
    config = AgenDevConfig(project_name="AgenDev Dashboard")
    agendev = AgenDev(config)

# Global variables for autonomous process
autonomous_thread = None
autonomous_running = False
user_responses = {}
user_questions = []

# Enhanced AgenDev methods for autonomy
def generate_tasks_from_description(project_name, project_description):
    """Generate epics and tasks based on project description using LLM."""
    logger.info(f"Generating tasks for project: {project_name}")
    
    # Use the LLM to analyze the project description and generate tasks
    try:
        # Create a system message for the LLM
        system_message = "You are an expert software architect and project planner. Your task is to analyze a project description and create a structured plan with epics and tasks."
        
        # Create prompt for the LLM
        prompt = f"""
        Project Name: {project_name}
        Project Description: {project_description}
        
        Based on this description, please create a comprehensive plan with:
        1. 3-7 high-level epics that represent major components or milestones
        2. 3-10 specific tasks for each epic
        
        For each epic, provide:
        - Title
        - Description
        - Priority (HIGH, MEDIUM, or LOW)
        - Risk level (HIGH, MEDIUM, or LOW)
        
        For each task, provide:
        - Title
        - Description
        - Type (IMPLEMENTATION, REFACTOR, BUGFIX, TEST, DOCUMENTATION, or PLANNING)
        - Priority (HIGH, MEDIUM, or LOW)
        - Risk level (HIGH, MEDIUM, or LOW)
        - Estimated duration in hours
        - Dependencies (if any, by task title)
        
        Structure your response as a JSON object.
        """
        
        # Define schema for structured output
        json_schema = {
            "name": "project_plan",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "epics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "risk": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "tasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "type": {"type": "string", "enum": ["IMPLEMENTATION", "REFACTOR", "BUGFIX", "TEST", "DOCUMENTATION", "PLANNING"]},
                                            "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                            "risk": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                            "estimated_duration_hours": {"type": "number"},
                                            "dependencies": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["title", "description", "type", "priority", "risk", "estimated_duration_hours"]
                                    }
                                }
                            },
                            "required": ["title", "description", "priority", "risk", "tasks"]
                        }
                    }
                },
                "required": ["epics"]
            }
        }
        
        # Make the request to the LLM
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=prompt)
        ]
        
        # Get structured response
        response = agendev.llm.llm_client.structured_completion(
            messages=messages,
            json_schema=json_schema,
            temperature=0.7
        )
        
        # Process the response to create epics and tasks
        epic_ids = []
        task_ids = []
        task_title_to_id = {}
        
        # Create epics
        for epic_data in response.get("epics", []):
            epic_id = agendev.create_epic(
                title=epic_data["title"],
                description=epic_data["description"],
                priority=TaskPriority(epic_data["priority"].lower()),
                risk=TaskRisk(epic_data["risk"].lower())
            )
            epic_ids.append(epic_id)
            
            # Create tasks for this epic
            for task_data in epic_data.get("tasks", []):
                task_id = agendev.create_task(
                    title=task_data["title"],
                    description=task_data["description"],
                    task_type=TaskType(task_data["type"].lower()),
                    priority=TaskPriority(task_data["priority"].lower()),
                    risk=TaskRisk(task_data["risk"].lower()),
                    estimated_duration_hours=task_data["estimated_duration_hours"],
                    epic_id=epic_id
                )
                task_ids.append(task_id)
                task_title_to_id[task_data["title"]] = task_id
        
        # Process dependencies (second pass)
        for epic_data in response.get("epics", []):
            for task_data in epic_data.get("tasks", []):
                if "dependencies" in task_data and task_data["dependencies"]:
                    task_id = task_title_to_id.get(task_data["title"])
                    if task_id:
                        for dep_title in task_data["dependencies"]:
                            dep_id = task_title_to_id.get(dep_title)
                            if dep_id and dep_id in agendev.task_graph.tasks:
                                # Add dependency relationship
                                agendev.task_graph.tasks[task_id].dependencies.append(dep_id)
                                agendev.task_graph.tasks[dep_id].dependents.append(task_id)
        
        # Update task statuses based on dependencies
        agendev.task_graph.update_task_statuses()
        
        # Notify about plan generation
        if agendev.notification_manager:
            agendev.notification_manager.success(
                f"Generated project plan with {len(epic_ids)} epics and {len(task_ids)} tasks."
            )
            
        return {
            "success": True,
            "epic_count": len(epic_ids),
            "task_count": len(task_ids),
            "epic_ids": [str(eid) for eid in epic_ids],
            "task_ids": [str(tid) for tid in task_ids]
        }
    
    except Exception as e:
        logger.error(f"Error generating tasks from description: {e}")
        if agendev.notification_manager:
            agendev.notification_manager.error(f"Failed to generate tasks: {e}")
        return {"success": False, "error": str(e)}

def autonomous_development_process():
    """Main function for autonomous development process."""
    global autonomous_running, user_questions, user_responses
    
    try:
        # Notify start
        if agendev.notification_manager:
            agendev.notification_manager.info("Starting autonomous development process...")
        
        # Step 1: Generate implementation plan
        logger.info("Generating implementation plan...")
        plan = agendev.generate_implementation_plan(max_iterations=500)
        
        if not plan:
            raise Exception("Failed to generate implementation plan")
            
        # Notify about plan generation
        if agendev.notification_manager:
            agendev.notification_manager.success(
                f"Implementation plan generated with {len(plan.task_sequence)} tasks."
            )
        
        # Step 2: Implement tasks in sequence
        for task_id in plan.task_sequence:
            # Check if we need to ask the user any questions before implementing this task
            task = agendev.task_graph.tasks.get(task_id)
            if not task:
                continue
                
            logger.info(f"Processing task: {task.title}")
            
            # Check if this is a complex task that might need clarification
            if task.risk in [TaskRisk.HIGH, TaskRisk.CRITICAL] or task.priority == TaskPriority.CRITICAL:
                # Generate a question for the user
                question = generate_question_for_task(task)
                if question:
                    # Add to questions queue
                    user_questions.append({
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "question": question,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Notify about question
                    if agendev.notification_manager:
                        agendev.notification_manager.info(
                            f"Question about task '{task.title}'. Please check the dashboard."
                        )
                    
                    # Wait for response
                    wait_time = 0
                    max_wait = 300  # 5 minutes max wait
                    while str(task_id) not in user_responses and wait_time < max_wait:
                        time.sleep(5)
                        wait_time += 5
                        if not autonomous_running:
                            return  # Exit if process was stopped
            
            # Implement the task
            try:
                # Check if we have user input to consider
                additional_context = user_responses.get(str(task_id), "")
                
                # Implement with additional context if available
                result = implement_task_with_context(task_id, additional_context)
                
                # Notify about implementation
                if agendev.notification_manager:
                    agendev.notification_manager.success(
                        f"Task '{task.title}' implemented successfully."
                    )
            except Exception as e:
                logger.error(f"Error implementing task {task.title}: {e}")
                if agendev.notification_manager:
                    agendev.notification_manager.error(f"Failed to implement task '{task.title}': {e}")
        
        # Notify about completion
        if agendev.notification_manager:
            agendev.notification_manager.milestone("Project implementation completed!")
            
        # Generate final summary
        summary = agendev.summarize_progress(voice_summary=True)
        
    except Exception as e:
        logger.error(f"Error in autonomous development process: {e}")
        if agendev.notification_manager:
            agendev.notification_manager.error(f"Autonomous development process failed: {e}")
    finally:
        autonomous_running = False

def generate_question_for_task(task):
    """Generate a question to ask the user about a complex task."""
    try:
        # Use LLM to generate a relevant question
        system_message = "You are an AI assistant helping with software development. Generate a specific question to ask the user about this task."
        
        prompt = f"""
        Task: {task.title}
        Description: {task.description}
        Priority: {task.priority.value}
        Risk: {task.risk.value}
        
        This task has been identified as complex or critical. Please generate a specific question to ask the user 
        that would help clarify requirements or provide guidance for implementing this task.
        
        The question should be focused, specific, and directly related to implementing this particular task.
        """
        
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=prompt)
        ]
        
        question = agendev.llm.llm_client.chat_completion(messages=messages, temperature=0.7)
        return question
    
    except Exception as e:
        logger.error(f"Error generating question for task: {e}")
        return None

def implement_task_with_context(task_id, additional_context=""):
    """Implement a task with additional context from user."""
    if task_id not in agendev.task_graph.tasks:
        raise Exception(f"Task not found: {task_id}")
    
    task = agendev.task_graph.tasks[task_id]
    
    # Create enhanced implementation with additional context
    if additional_context:
        # Add the additional context to the task description
        enhanced_task = task.model_copy()
        enhanced_task.description = f"{task.description}\n\nAdditional context from user: {additional_context}"
        
        # Store the original task
        original_task = agendev.task_graph.tasks[task_id]
        
        # Temporarily replace with enhanced task
        agendev.task_graph.tasks[task_id] = enhanced_task
        
        # Implement the task
        result = agendev.implement_task(task_id)
        
        # Restore original task (but keep the status and artifacts)
        original_task.status = enhanced_task.status
        original_task.completion_percentage = enhanced_task.completion_percentage
        original_task.actual_duration_hours = enhanced_task.actual_duration_hours
        original_task.artifact_paths = enhanced_task.artifact_paths
        agendev.task_graph.tasks[task_id] = original_task
    else:
        # Implement normally
        result = agendev.implement_task(task_id)
    
    return result

def start_autonomous_process():
    """Start the autonomous development process in a separate thread."""
    global autonomous_thread, autonomous_running
    
    if autonomous_running:
        return {"success": False, "message": "Autonomous process already running"}
    
    autonomous_running = True
    autonomous_thread = threading.Thread(target=autonomous_development_process)
    autonomous_thread.daemon = True
    autonomous_thread.start()
    
    return {"success": True, "message": "Autonomous process started"}

def stop_autonomous_process():
    """Stop the autonomous development process."""
    global autonomous_running
    autonomous_running = False
    return {"success": True, "message": "Autonomous process stopping"}

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
    card_props = {
        "children": [
            dbc.CardHeader(
                html.H5(title, className="mb-0"),
                className="bg-transparent border-bottom border-dark"
            ),
            dbc.CardBody(content)
        ],
        "className": "shadow-sm mb-4"
    }
    
    # Only add id if it's not None
    if id is not None:
        card_props["id"] = id
        
    return dbc.Card(**card_props)

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
                [html.I(className="fas fa-question-circle me-2"), "Information Requests"],
                href="#",
                id="nav-requests"
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
                className="mb-3",
                id="project-status-badge"
            ),
        ]),
        html.Div([
            html.P("Autonomous Process"),
            dbc.Badge(
                "Inactive", 
                color="secondary", 
                className="mb-3",
                id="autonomous-status-badge"
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

# Project creation form for the overview page
def create_project_form():
    """Create the project creation form."""
    return html.Div([
        html.H3("Create New Project", className="mb-4"),
        dbc.Form([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project Name"),
                    dbc.Input(id="project-name", type="text", placeholder="Enter project name"),
                ], md=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project Description"),
                    dbc.Textarea(
                        id="project-description", 
                        placeholder="Describe your project in detail...",
                        style={"height": "200px"}
                    ),
                ]),
            ], className="mt-3"),
            dbc.Button(
                [html.I(className="fas fa-rocket me-2"), "Create & Start Project"], 
                id="btn-create-project", 
                color="primary",
                size="lg",
                className="mt-4 w-100",
            ),
            html.Div(id="project-creation-output", className="mt-3")
        ]),
    ])

# Create pages
def create_overview_page():
    """Generate the overview page content"""
    project_status = agendev.get_project_status()
    progress = project_status.get('progress', {}).get('percentage', 0)
    
    # Check if project has started (has tasks)
    has_project = project_status.get('tasks', {}).get('total', 0) > 0
    
    if not has_project:
        # Show project creation form
        return dbc.Container([
            html.H2("Welcome to AgenDev", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    create_card(
                        "Autonomous Development System",
                        html.P("Enter your project details to start. AgenDev will automatically "
                               "plan and implement your project, asking for clarification only when needed.")
                    ),
                ], md=5),
                dbc.Col([
                    create_card("Create New Project", create_project_form()),
                ], md=7),
            ]),
        ])
    else:
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
                dbc.Button(
                    [
                        html.I(className="fas fa-play me-2"), 
                        "Start Autonomous Process"
                    ], 
                    id="btn-start-auto", 
                    color="success",
                    className="me-2"
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-stop me-2"), 
                        "Stop Autonomous Process"
                    ], 
                    id="btn-stop-auto", 
                    color="danger"
                ),
                html.Div(id="auto-process-output", className="mt-3")
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
    
    # Extract epic data for the table
    epics_data = []
    for epic_id, epic in agendev.task_graph.epics.items():
        epics_data.append({
            "id": str(epic_id),
            "title": epic.title,
            "status": epic.status.value,
            "priority": epic.priority.value,
            "risk": epic.risk.value,
            "progress": f"{epic.milestone_percentage:.1f}%",
        })
    
    # Create tasks table
    if tasks_data:
        tasks_table = dbc.Table.from_dataframe(
            pd.DataFrame(tasks_data)[["title", "status", "type", "priority", "risk", "duration"]],
            striped=True,
            bordered=False,
            hover=True,
            className="table-dark",
        )
    else:
        tasks_table = html.Div(
            html.P("No tasks found. Create a project to generate tasks automatically."),
            className="text-center p-5"
        )
    
    # Create epics table
    if epics_data:
        epics_table = dbc.Table.from_dataframe(
            pd.DataFrame(epics_data)[["title", "status", "priority", "risk", "progress"]],
            striped=True,
            bordered=False,
            hover=True,
            className="table-dark",
        )
    else:
        epics_table = html.Div(
            html.P("No epics found. Create a project to generate epics automatically."),
            className="text-center p-5"
        )
    
    return dbc.Container([
        html.H2("Tasks & Epics", className="mb-4"),
        dbc.Row([
            dbc.Col([
                create_card("Epics", epics_table),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                create_card("Tasks", tasks_table),
            ]),
        ]),
    ])

def create_implementation_page():
    """Generate the implementation page content"""
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
    
    # Implementation status
    implementation_status = []
    for task_id, task in agendev.task_graph.tasks.items():
        status_color = "secondary"
        if task.status == TaskStatus.COMPLETED:
            status_color = "success"
        elif task.status == TaskStatus.IN_PROGRESS:
            status_color = "primary"
        elif task.status == TaskStatus.BLOCKED:
            status_color = "warning"
        elif task.status == TaskStatus.FAILED:
            status_color = "danger"
            
        implementation_status.append(
            html.Div([
                dbc.Badge(task.status.value, color=status_color, className="me-2"),
                html.Span(task.title),
                html.Small(f" ({task.task_type.value})", className="text-muted ms-1")
            ], className="mb-2")
        )
    
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
        code_content = html.P("No implementations available yet. Start the autonomous process to generate implementations.")
    
    # Create file selector
    file_options = []
    for impl in implementations:
        file_options.append({
            "label": impl["file_name"],
            "value": impl["task_id"]
        })
    
    file_selector = html.Div([
        dbc.Label("Select Implementation"),
        dcc.Dropdown(
            id="implementation-file",
            options=file_options,
            value=file_options[0]["value"] if file_options else None,
            className="mb-3",
        )
    ]) if file_options else html.P("No files available yet.")
    
    return dbc.Container([
        html.H2("Implementation", className="mb-4"),
        dbc.Row([
            dbc.Col(create_card("Implementation Status", html.Div(implementation_status)), md=4),
            dbc.Col([
                create_card("Implementation Files", file_selector),
                create_card("Code Viewer", html.Div(code_content, id="code-viewer")),
            ], md=8),
        ]),
    ])

def create_requests_page():
    """Generate the information requests page"""
    global user_questions
    
    # Display information requests
    requests_content = []
    
    if not user_questions:
        requests_content.append(
            html.P("No information requests at this time. The system will ask questions here when needed.")
        )
    else:
        for question in user_questions:
            # Check if this question has been answered
            is_answered = question["task_id"] in user_responses
            
            requests_content.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H6(f"Question about: {question['task_title']}"),
                        html.Small(question["timestamp"], className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.P(question["question"]),
                        dbc.Textarea(
                            id={"type": "question-response", "index": question["task_id"]},
                            placeholder="Type your response here...",
                            disabled=is_answered,
                            value=user_responses.get(question["task_id"], ""),
                            style={"height": "100px"}
                        ),
                        dbc.Button(
                            "Submit Response", 
                            id={"type": "question-submit", "index": question["task_id"]},
                            color="primary",
                            className="mt-2",
                            disabled=is_answered
                        ),
                        html.Div(
                            id={"type": "question-result", "index": question["task_id"]}
                        )
                    ])
                ], className="mb-4", color="dark")
            )
    
    return dbc.Container([
        html.H2("Information Requests", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.P("When the system needs additional information to continue development, "
                       "questions will appear here for you to answer."),
                html.Div(requests_content),
            ]),
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
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 seconds in milliseconds
            n_intervals=0
        ),
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
        Input("nav-requests", "n_clicks"),
    ],
    [State("active-page", "data")],
    prevent_initial_call=True,
)
def update_active_page(overview_clicks, tasks_clicks, implementation_clicks, requests_clicks, current):
    ctx = callback_context
    if not ctx.triggered:
        return current
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    page_map = {
        "nav-overview": "overview",
        "nav-tasks": "tasks",
        "nav-implementation": "implementation",
        "nav-requests": "requests",
    }
    return page_map.get(button_id, current)

@callback(
    [
        Output("nav-overview", "active"),
        Output("nav-tasks", "active"),
        Output("nav-implementation", "active"),
        Output("nav-requests", "active"),
    ],
    Input("active-page", "data"),
)
def update_nav_active(active_page):
    return (
        active_page == "overview",
        active_page == "tasks",
        active_page == "implementation", 
        active_page == "requests",
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
    elif active_page == "requests":
        return create_requests_page()
    else:
        return create_overview_page()

# Callback for project creation
@callback(
    Output("project-creation-output", "children"),
    Input("btn-create-project", "n_clicks"),
    [
        State("project-name", "value"),
        State("project-description", "value"),
    ],
    prevent_initial_call=True,
)
def handle_create_project(n_clicks, project_name, project_description):
    if not project_name or not project_description:
        return dbc.Alert("Project name and description are required", color="danger")
    
    try:
        # Update project name
        agendev.config.project_name = project_name
        
        # Generate tasks and epics based on description
        result = generate_tasks_from_description(project_name, project_description)
        
        if not result.get("success", False):
            return dbc.Alert(f"Error generating project plan: {result.get('error', 'Unknown error')}", color="danger")
        
        # Generate implementation plan
        agendev.generate_implementation_plan()
        
        # Start autonomous process
        start_result = start_autonomous_process()
        
        return dbc.Alert(
            [
                html.P(f"Project '{project_name}' created successfully with {result['epic_count']} epics and {result['task_count']} tasks."),
                html.P("Autonomous development process has started! The system will now implement your project automatically.")
            ], 
            color="success"
        )
    except Exception as e:
        return dbc.Alert(f"Error creating project: {str(e)}", color="danger")

# Callback for autonomous process control
@callback(
    Output("auto-process-output", "children"),
    [
        Input("btn-start-auto", "n_clicks"),
        Input("btn-stop-auto", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_auto_process(start_clicks, stop_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-start-auto":
        result = start_autonomous_process()
        color = "success" if result.get("success", False) else "danger"
        return dbc.Alert(result.get("message", ""), color=color)
    elif button_id == "btn-stop-auto":
        result = stop_autonomous_process()
        color = "success" if result.get("success", False) else "danger"
        return dbc.Alert(result.get("message", ""), color=color)
    
    return ""

# Callback for voice summary
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

# Callback for file selection
@callback(
    Output("code-viewer", "children"),
    Input("implementation-file", "value"),
    prevent_initial_call=True,
)
def update_code_viewer(task_id):
    if not task_id:
        return html.P("No file selected")
    
    task = agendev.task_graph.tasks.get(UUID(task_id))
    if not task or not task.artifact_paths:
        return html.P("No artifact found for this task")
    
    try:
        file_path = resolve_path(task.artifact_paths[0])
        if not file_path.exists():
            return html.P(f"File not found: {task.artifact_paths[0]}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return html.Div([
            html.H6(Path(task.artifact_paths[0]).name, className="mb-3"),
            dbc.Card(
                dbc.CardBody(
                    dcc.Markdown(
                        f"```python\n{content}\n```",
                        className="mb-0",
                    )
                ),
                className="bg-dark"
            )
        ])
    except Exception as e:
        return html.P(f"Error loading file: {e}")

# Callback for question responses
@callback(
    Output({"type": "question-result", "index": dash.dependencies.MATCH}, "children"),
    Input({"type": "question-submit", "index": dash.dependencies.MATCH}, "n_clicks"),
    [
        State({"type": "question-response", "index": dash.dependencies.MATCH}, "value"),
        State({"type": "question-submit", "index": dash.dependencies.MATCH}, "id"),
    ],
    prevent_initial_call=True,
)
def handle_question_response(n_clicks, response, id_data):
    global user_responses
    
    if not response:
        return dbc.Alert("Please provide a response", color="danger")
    
    # Get the task ID from the ID
    task_id = id_data["index"]
    
    # Store the response
    user_responses[task_id] = response
    
    return dbc.Alert("Response submitted. The system will continue with implementation.", color="success")

# Callback to update status badges
@callback(
    [
        Output("autonomous-status-badge", "children"),
        Output("autonomous-status-badge", "color"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_status_badges(n_intervals):
    global autonomous_running
    
    if autonomous_running:
        return "Active", "success"
    else:
        return "Inactive", "secondary"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)