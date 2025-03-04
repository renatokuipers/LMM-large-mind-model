import dash
from dash import callback, Input, Output, State, html, dcc, no_update, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import json
import os
import subprocess
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid

# Import app from app.py
from ui.app import (
    app, create_projects_layout, create_new_project_layout, 
    create_project_detail_layout, create_task_detail_modal,
    create_loading_screen, create_error_page, create_code_preview_modal
)

# Import models and services
from models.project_model import Project, Epic, Task, ProjectStore, TaskStatus
from llm_module import LLMClient, Message
from services.planner import PlannerService
from services.code_generator import CodeGeneratorService

# Initialize services
llm_client = LLMClient()
planner_service = PlannerService(llm_client)
code_generator_service = CodeGeneratorService(llm_client)

# Storage file for projects
PROJECTS_FILE = "projects_data.json"

# Helper functions
def load_projects() -> ProjectStore:
    """Load projects from file"""
    try:
        return ProjectStore.load_from_file(PROJECTS_FILE)
    except Exception as e:
        print(f"Error loading projects: {str(e)}")
        return ProjectStore()

def save_projects(projects: ProjectStore) -> None:
    """Save projects to file"""
    try:
        projects.save_to_file(PROJECTS_FILE)
    except Exception as e:
        print(f"Error saving projects: {str(e)}")

def create_notification(message: str, title: str = "Notification", color: str = "primary", 
                     icon: str = "info-circle", duration: int = 4000) -> Dict:
    """Create a notification dictionary for the toast component"""
    return {
        "message": message,
        "title": title,
        "color": color,
        "icon": icon,
        "duration": duration,
        "is_open": True,
        "id": str(uuid.uuid4())
    }

def handle_error(e: Exception, default_message: str = "An error occurred") -> Tuple[str, str]:
    """Handle exceptions and return user-friendly error messages with details"""
    error_message = str(e)
    error_details = traceback.format_exc()
    
    # Make error message more user-friendly
    if "no such file or directory" in error_message.lower():
        user_message = "Could not access the specified directory. Please check that the path exists."
    elif "permission denied" in error_message.lower():
        user_message = "Permission denied. Please check your access rights."
    elif "invalid json" in error_message.lower() or "json decode" in error_message.lower():
        user_message = "Invalid data format. The project file might be corrupted."
    else:
        user_message = default_message
    
    print(f"Error: {error_message}")
    print(f"Details: {error_details}")
    
    return user_message, error_details

# TOAST NOTIFICATION CALLBACKS

# Display toast notification
@callback(
    Output("notification-toast", "is_open"),
    Output("notification-toast", "header"),
    Output("notification-toast", "children"),
    Output("notification-toast", "icon"),
    Output("notification-toast", "duration"),
    Input("toast-message", "data"),
    prevent_initial_call=True
)
def show_toast(toast_data):
    """Show toast notification with the provided data"""
    if not toast_data:
        return False, no_update, no_update, no_update, no_update
    
    icon_class = f"bi-{toast_data.get('icon', 'info-circle')}"
    
    return (
        True,
        toast_data.get("title", "Notification"),
        toast_data.get("message", ""),
        {"color": toast_data.get("color", "primary"), "className": f"bi {icon_class} me-2"},
        toast_data.get("duration", 4000)
    )

# FORM VALIDATION CALLBACKS

# Validate project form
@callback(
    [
        Output("project-name", "valid"),
        Output("project-name", "invalid"),
        Output("project-description", "valid"),
        Output("project-description", "invalid"),
        Output("output-directory", "valid"),
        Output("output-directory", "invalid"),
        Output("form-validation", "data"),
    ],
    [
        Input("btn-submit-project", "n_clicks"),
        Input("project-name", "value"),
        Input("project-description", "value"),
        Input("output-directory", "value"),
    ],
    [
        State("form-validation", "data"),
    ],
)
def validate_project_form(n_clicks, name, description, output_directory, validation_state):
    """Validate project creation form inputs"""
    # Initialize validation state
    if validation_state is None:
        validation_state = {"valid": False, "errors": []}
    
    # Only perform full validation when submit button is clicked
    validate_all = n_clicks and n_clicks > 0
    triggered_id = ctx.triggered_id
    
    # Initialize outputs (name_valid, name_invalid, desc_valid, desc_invalid, dir_valid, dir_invalid, validation_state)
    outputs = [False, False, False, False, False, False, validation_state]
    
    # Validate name
    name_valid = name and len(name) >= 3 and len(name) <= 100
    
    # Validate description
    desc_valid = description and len(description) >= 10
    
    # Validate output directory
    dir_valid = output_directory and len(output_directory) > 0
    try:
        if output_directory and triggered_id == "output-directory":
            os.makedirs(output_directory, exist_ok=True)
            dir_valid = True
    except Exception:
        dir_valid = False
    
    # Update specific field based on which input triggered the callback
    if triggered_id == "project-name":
        outputs[0] = name_valid  # name_valid
        outputs[1] = bool(name) and not name_valid  # name_invalid
    elif triggered_id == "project-description":
        outputs[2] = desc_valid  # desc_valid
        outputs[3] = bool(description) and not desc_valid  # desc_invalid
    elif triggered_id == "output-directory":
        outputs[4] = dir_valid  # dir_valid
        outputs[5] = bool(output_directory) and not dir_valid  # dir_invalid
    
    # For complete validation when submit is clicked
    if validate_all:
        outputs[0] = name_valid  # name_valid
        outputs[1] = not name_valid  # name_invalid
        outputs[2] = desc_valid  # desc_valid
        outputs[3] = not desc_valid  # desc_invalid
        outputs[4] = dir_valid  # dir_valid
        outputs[5] = not dir_valid  # dir_invalid
        
        # Update validation state
        errors = []
        if not name_valid:
            errors.append("Project name must be between 3 and 100 characters")
        if not desc_valid:
            errors.append("Project description must be at least 10 characters")
        if not dir_valid:
            errors.append("Please provide a valid output directory")
        
        outputs[6] = {"valid": all([name_valid, desc_valid, dir_valid]), "errors": errors}
    
    return outputs

# Display form errors
@callback(
    Output("form-error-message", "children"),
    Input("form-validation", "data"),
    prevent_initial_call=True
)
def show_form_errors(validation_data):
    """Display form validation errors"""
    if not validation_data or not validation_data.get("errors"):
        return ""
    
    errors = validation_data.get("errors", [])
    if not errors:
        return ""
    
    return [
        html.I(className="bi bi-exclamation-triangle-fill me-2"),
        "Please fix the following errors:",
        html.Ul([
            html.Li(error) for error in errors
        ], className="mb-0 mt-2")
    ]

# NAVIGATION CALLBACKS

# Navigate to projects list
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [Input("nav-projects", "n_clicks")],
    [State("projects-store", "data")],
    prevent_initial_call=True
)
def nav_to_projects(n_clicks, projects_data):
    """Navigate to projects list"""
    try:
        projects = load_projects()
        projects_dict = json.loads(projects.json()) if projects else {"projects": {}}
        return create_projects_layout(projects_dict), no_update
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading projects")
        return create_error_page(
            "Error Loading Projects", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Navigate to create new project form
@callback(
    Output("page-content", "children", allow_duplicate=True),
    [Input("nav-create", "n_clicks")],
    prevent_initial_call=True
)
def nav_to_create(n_clicks):
    """Navigate to create new project form"""
    return create_new_project_layout()

# Navigate to create project form from button
@callback(
    Output("page-content", "children", allow_duplicate=True),
    [Input("btn-create-project", "n_clicks")],
    prevent_initial_call=True
)
def create_button_to_form(n_clicks):
    """Navigate to create project form"""
    return create_new_project_layout()

# Navigate back to projects list
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [Input("btn-back-to-projects", "n_clicks")],
    [State("projects-store", "data")],
    prevent_initial_call=True
)
def back_to_projects(n_clicks, projects_data):
    """Navigate back to projects list"""
    try:
        projects = load_projects()
        projects_dict = json.loads(projects.json()) if projects else {"projects": {}}
        return create_projects_layout(projects_dict), no_update
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading projects")
        return create_error_page(
            "Error Loading Projects", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Cancel project creation and return to projects list
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [Input("btn-cancel-project", "n_clicks")],
    [State("projects-store", "data")],
    prevent_initial_call=True
)
def cancel_project(n_clicks, projects_data):
    """Cancel project creation and return to projects list"""
    try:
        projects = load_projects()
        projects_dict = json.loads(projects.json()) if projects else {"projects": {}}
        return create_projects_layout(projects_dict), create_notification(
            "Project creation cancelled", 
            "Cancelled", 
            "secondary", 
            "x-circle"
        )
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading projects")
        return create_error_page(
            "Error Loading Projects", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# View project detail
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("current-project-id", "data", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [Input({"type": "btn-view-project", "index": ALL}, "n_clicks")],
    [
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def view_project(btn_clicks, projects_data):
    """Navigate to project detail view"""
    if not any(btn_clicks):
        return no_update, no_update, no_update
    
    try:
        ctx_triggered = ctx.triggered_id
        
        if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "btn-view-project":
            project_id = ctx_triggered.get("index")
            project_data = projects_data.get("projects", {}).get(project_id)
            
            if project_data:
                return create_project_detail_layout(project_id, project_data), project_id, no_update
            else:
                return create_error_page(
                    "Project Not Found", 
                    "The requested project could not be found.",
                    None
                ), no_update, create_notification(
                    "Project not found", 
                    "Error", 
                    "warning", 
                    "question-circle"
                )
        
        return no_update, no_update, no_update
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading project details")
        return create_error_page(
            "Error Loading Project", 
            error_msg,
            error_details
        ), no_update, create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Project card hover effect
@callback(
    Output({"type": "project-card", "index": MATCH}, "style"),
    [
        Input({"type": "project-card", "index": MATCH}, "n_hover"),
    ],
    [
        State({"type": "project-card", "index": MATCH}, "style"),
    ],
    prevent_initial_call=True
)
def project_card_hover(n_hover, current_style):
    """Add hover effect to project cards"""
    from ui.app import CARD_STYLE, HOVER_STYLE
    
    if n_hover and n_hover % 2 == 1:  # Mouse enter (odd number of hovers)
        return {**CARD_STYLE, **HOVER_STYLE}
    else:  # Mouse leave (even number of hovers)
        return CARD_STYLE

# PROJECT CREATION AND PLANNING CALLBACKS

# Project creation callback
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("current-project-id", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("btn-submit-project", "n_clicks"),
    ],
    [
        State("project-name", "value"),
        State("project-description", "value"),
        State("output-directory", "value"),
        State("projects-store", "data"),
        State("form-validation", "data"),
    ],
    prevent_initial_call=True
)
def create_project(n_clicks, name, description, output_directory, existing_projects_data, validation_data):
    """Create a new project and generate its plan"""
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    
    # Check form validation
    if not validation_data or not validation_data.get("valid", False):
        return no_update, no_update, no_update, create_notification(
            "Please fix the form errors before submitting", 
            "Validation Error", 
            "warning", 
            "exclamation-triangle"
        )
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Create project object
        project = Project(
            name=name,
            description=description,
            output_directory=output_directory
        )
        
        # Add project to store
        projects = load_projects()
        projects.add_project(project)
        save_projects(projects)
        
        # Convert to dictionary
        projects_dict = json.loads(projects.json())
        
        # Show loading screen
        loading_layout = create_loading_screen(
            f"Creating Project: {name}",
            "Generating project plan... This may take a few moments."
        )
        
        return projects_dict, project.id, loading_layout, create_notification(
            f"Project '{name}' created. Generating plan...", 
            "Project Created", 
            "success", 
            "check-circle"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error creating project")
        return no_update, no_update, create_error_page(
            "Error Creating Project", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Project planning process
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
        Output("loading-status", "children", allow_duplicate=True),
        Output("loading-progress", "value", allow_duplicate=True),
    ],
    [
        Input("current-project-id", "data"),
        Input("loading-status", "children"),
    ],
    [
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def plan_project(project_id, loading_status, existing_projects_data):
    """Generate plan for the newly created project with progress updates"""
    # Skip if no project ID or if this was triggered by a loading status update
    if not project_id or ctx.triggered_id == "loading-status":
        return no_update, no_update, no_update, no_update, no_update
    
    try:
        # Load existing projects
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            # Check if it's a newly created project
            if existing_projects_data and existing_projects_data.get("projects", {}).get(project_id):
                project_data = existing_projects_data.get("projects", {}).get(project_id)
                project = Project.parse_obj(project_data)
            else:
                return no_update, create_error_page(
                    "Project Not Found", 
                    "The project could not be found.",
                    None
                ), create_notification(
                    "Project not found", 
                    "Error", 
                    "warning", 
                    "exclamation-triangle"
                ), no_update, no_update
        
        # Update loading status
        return no_update, no_update, no_update, "Starting project planning...", 10
    
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error planning project")
        return no_update, create_error_page(
            "Error Planning Project", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        ), no_update, no_update

# Project planning execution
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("loading-progress", "value"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
        State("loading-status", "children"),
    ],
    prevent_initial_call=True
)
def execute_plan_project(progress_value, project_id, existing_projects_data, loading_status):
    """Execute project planning after loading screen appears"""
    if not progress_value or progress_value != 10:
        return no_update, no_update, no_update
    
    try:
        # Load existing projects
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            # Check if it's a newly created project
            if existing_projects_data and existing_projects_data.get("projects", {}).get(project_id):
                project_data = existing_projects_data.get("projects", {}).get(project_id)
                project = Project.parse_obj(project_data)
            else:
                return no_update, create_error_page(
                    "Project Not Found", 
                    "The project could not be found.",
                    None
                ), create_notification(
                    "Project not found", 
                    "Error", 
                    "warning", 
                    "exclamation-triangle"
                )
        
        # Generate project plan
        project = planner_service.create_project_plan(project)
        
        # Save updated project
        projects.add_project(project)
        save_projects(projects)
        
        # Convert to dictionary for storage
        projects_dict = json.loads(projects.json())
        
        # Return to project detail view
        project_dict = projects_dict.get("projects", {}).get(project_id)
        return projects_dict, create_project_detail_layout(project_id, project_dict), create_notification(
            f"Project plan for '{project.name}' has been generated", 
            "Plan Generated", 
            "success", 
            "check-circle"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error planning project")
        return no_update, create_error_page(
            "Error Planning Project", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# TASK AND CODE MANAGEMENT CALLBACKS

# Task detail modal
@callback(
    Output("task-detail-modal", "is_open"),
    [
        Input({"type": "btn-task-details", "epic": ALL, "task": ALL}, "n_clicks"),
        Input("close-task-modal", "n_clicks"),
    ],
    prevent_initial_call=True
)
def toggle_task_modal(task_btn_clicks, close_click):
    """Open or close the task detail modal"""
    if ctx.triggered_id == "close-task-modal":
        return False
    
    if any(click for click in task_btn_clicks if click):
        return True
    
    return no_update

# Populate task detail modal
@callback(
    [
        Output("task-detail-modal", "children"),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input({"type": "btn-task-details", "epic": ALL, "task": ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def show_task_details(btn_clicks, project_id, projects_data):
    """Show task details in modal"""
    if not any(click for click in btn_clicks if click):
        return no_update, no_update
    
    try:
        # Get the triggered button
        triggered_idx = [i for i, n in enumerate(btn_clicks) if n]
        if not triggered_idx:
            return no_update, no_update
        
        # Get the triggered context
        triggered = ctx.triggered[0]
        triggered_id = ctx.triggered_id
        
        if isinstance(triggered_id, list):
            triggered_id = triggered_id[triggered_idx[0]]
        
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "btn-task-details":
            return no_update, no_update
        
        epic_id = triggered_id.get("epic")
        task_id = triggered_id.get("task")
        
        project_data = projects_data.get("projects", {}).get(project_id)
        
        if not project_data:
            return dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Project data not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
            ], id="task-detail-modal"), create_notification(
                "Project data not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        return create_task_detail_modal(epic_id, task_id, project_data), no_update
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading task details")
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody(f"Error loading task details: {error_msg}"),
            dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
        ], id="task-detail-modal"), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Generate code for a specific task
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("task-detail-modal", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input({"type": "btn-generate-task", "epic": ALL, "task": ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def generate_task_code(btn_clicks, project_id, projects_data):
    """Generate code for a specific task"""
    if not any(click for click in btn_clicks if click):
        return no_update, no_update, no_update
    
    try:
        # Get the triggered button
        triggered_idx = [i for i, n in enumerate(btn_clicks) if n]
        if not triggered_idx:
            return no_update, no_update, no_update
        
        # Get the triggered context
        triggered = ctx.triggered[0]
        triggered_id = ctx.triggered_id
        
        if isinstance(triggered_id, list):
            triggered_id = triggered_id[triggered_idx[0]]
        
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "btn-generate-task":
            return no_update, no_update, no_update
        
        epic_id = triggered_id.get("epic")
        task_id = triggered_id.get("task")
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Project not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
            ], id="task-detail-modal"), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Find epic and task
        target_epic = None
        target_task = None
        
        for epic in project.epics:
            if epic.id == epic_id:
                target_epic = epic
                for task in epic.tasks:
                    if task.id == task_id:
                        target_task = task
                        break
                break
        
        if not target_epic or not target_task:
            return no_update, dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Epic or task not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
            ], id="task-detail-modal"), create_notification(
                "Epic or task not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Generate loading modal
        loading_modal = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(f"Generating Code: {target_task.title}")),
            dbc.ModalBody([
                dbc.Spinner(size="lg", color="primary", type="grow", className="mb-3 d-block mx-auto"),
                dbc.Progress(
                    value=40, 
                    striped=True, 
                    animated=True,
                    className="mb-3"
                ),
                html.P("Generating code implementation...", className="text-center"),
                html.P("This may take a few moments...", className="text-center text-muted"),
            ]),
        ], id="task-detail-modal", is_open=True)
        
        # First update UI to show loading
        return no_update, loading_modal, create_notification(
            f"Generating code for task: {target_task.title}", 
            "Code Generation", 
            "info", 
            "code-slash"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error initiating code generation")
        return no_update, dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody(f"Error initiating code generation: {error_msg}"),
            dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
        ], id="task-detail-modal"), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Execute task code generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("task-detail-modal", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("task-detail-modal", "is_open"),
    ],
    [
        State("task-detail-modal", "children"),
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def execute_task_code_generation(is_open, modal_children, project_id, projects_data):
    """Execute code generation after showing loading modal"""
    if not is_open or not isinstance(modal_children, list):
        return no_update, no_update, no_update
    
    # Check if we're in loading state by looking for "Generating Code" in the header
    loading_state = False
    for child in modal_children:
        if child['type'] == 'ModalHeader':
            for header_child in child['props']['children']['props']['children']:
                if isinstance(header_child, str) and "Generating Code" in header_child:
                    loading_state = True
    
    if not loading_state:
        return no_update, no_update, no_update
    
    try:
        # Get the epic and task IDs from the current triggered buttons
        epic_id = None
        task_id = None
        
        # Find all buttons in the UI
        for button_id in ctx.inputs_list:
            if isinstance(button_id, dict) and button_id.get('type') == 'btn-generate-task':
                epic_id = button_id.get('epic')
                task_id = button_id.get('task')
                break
        
        if not epic_id or not task_id:
            return no_update, dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Could not determine which task to generate code for"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
            ], id="task-detail-modal"), create_notification(
                "Code generation failed - could not identify task", 
                "Error", 
                "danger", 
                "exclamation-triangle"
            )
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            # Try getting from the projects_data store
            if projects_data and projects_data.get("projects", {}).get(project_id):
                project_data = projects_data.get("projects", {}).get(project_id)
                project = Project.parse_obj(project_data)
            else:
                return no_update, dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Error")),
                    dbc.ModalBody("Project not found"),
                    dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
                ], id="task-detail-modal"), create_notification(
                    "Project not found", 
                    "Error", 
                    "warning", 
                    "exclamation-triangle"
                )
        
        # Find epic and task
        target_epic = None
        target_task = None
        
        for epic in project.epics:
            if epic.id == epic_id:
                target_epic = epic
                for task in epic.tasks:
                    if task.id == task_id:
                        target_task = task
                        break
                break
        
        if not target_epic or not target_task:
            return no_update, dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Epic or task not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
            ], id="task-detail-modal"), create_notification(
                "Epic or task not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Generate code
        updated_task = code_generator_service.generate_code_for_task(project, target_epic, target_task)
        
        # Update project
        for i, epic in enumerate(project.epics):
            if epic.id == epic_id:
                for j, task in enumerate(epic.tasks):
                    if task.id == task_id:
                        project.epics[i].tasks[j] = updated_task
                        break
                break
        
        # Save updated project
        projects.add_project(project)
        save_projects(projects)
        
        # Convert to dictionary for storage
        projects_dict = json.loads(projects.json())
        
        # Return updated modal
        project_dict = projects_dict.get("projects", {}).get(project_id)
        success = updated_task.status == TaskStatus.COMPLETED
        
        return projects_dict, create_task_detail_modal(epic_id, task_id, project_dict), create_notification(
            f"Code generation {'completed successfully' if success else 'failed'} for task: {updated_task.title}", 
            "Code Generation", 
            "success" if success else "warning", 
            "check-circle" if success else "exclamation-triangle"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error generating code")
        return no_update, dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody(f"Error generating code: {error_msg}"),
            dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
        ], id="task-detail-modal"), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Code preview modal
@callback(
    Output("code-preview-modal", "children"),
    [
        Input({"type": "btn-view-code", "epic": ALL, "task": ALL, "item": ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def show_code_preview(btn_clicks, project_id, projects_data):
    """Show code preview in modal"""
    if not any(click for click in btn_clicks if click):
        return no_update
    
    try:
        # Get the triggered button
        triggered_idx = [i for i, n in enumerate(btn_clicks) if n]
        if not triggered_idx:
            return no_update
        
        # Get the triggered context
        triggered = ctx.triggered[0]
        triggered_id = ctx.triggered_id
        
        if isinstance(triggered_id, list):
            triggered_id = triggered_id[triggered_idx[0]]
        
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "btn-view-code":
            return no_update
        
        epic_id = triggered_id.get("epic")
        task_id = triggered_id.get("task")
        item_index = triggered_id.get("item")
        
        project_data = projects_data.get("projects", {}).get(project_id)
        
        if not project_data:
            return dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("Project data not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-code-preview", className="ms-auto")),
            ], id="code-preview-modal")
        
        return create_code_preview_modal(epic_id, task_id, item_index, project_data)
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error loading code preview")
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody(f"Error loading code preview: {error_msg}"),
            dbc.ModalFooter(dbc.Button("Close", id="close-code-preview", className="ms-auto")),
        ], id="code-preview-modal")

# Toggle code preview modal
@callback(
    Output("code-preview-modal", "is_open"),
    [
        Input({"type": "btn-view-code", "epic": ALL, "task": ALL, "item": ALL}, "n_clicks"),
        Input("close-code-preview", "n_clicks"),
    ],
    prevent_initial_call=True
)
def toggle_code_preview_modal(btn_clicks, close_click):
    """Open or close the code preview modal"""
    if ctx.triggered_id == "close-code-preview":
        return False
    
    if any(click for click in btn_clicks if click):
        return True
    
    return no_update

# Generate code for an entire epic
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input({"type": "btn-generate-epic", "index": ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def generate_epic_code(btn_clicks, project_id, projects_data):
    """Generate code for all tasks in an epic"""
    if not any(click for click in btn_clicks if click):
        return no_update, no_update
    
    try:
        # Get the triggered button
        triggered_idx = [i for i, n in enumerate(btn_clicks) if n]
        if not triggered_idx:
            return no_update, no_update
        
        # Get the triggered context
        triggered = ctx.triggered[0]
        triggered_id = ctx.triggered_id
        
        if isinstance(triggered_id, list):
            triggered_id = triggered_id[triggered_idx[0]]
        
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "btn-generate-epic":
            return no_update, no_update
        
        epic_id = triggered_id.get("index")
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return create_error_page(
                "Project Not Found", 
                "The project could not be found.",
                None
            ), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Find epic
        target_epic = None
        for epic in project.epics:
            if epic.id == epic_id:
                target_epic = epic
                break
        
        if not target_epic:
            return create_error_page(
                "Epic Not Found", 
                "The requested epic could not be found.",
                None
            ), create_notification(
                "Epic not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Generate loading screen
        loading_layout = create_loading_screen(
            f"Generating Code for Epic: {target_epic.title}",
            "Implementing all tasks in this epic... This may take a few moments."
        )
        
        # Store the epic ID for the next callback
        global_epic_id = epic_id  # Not ideal, but works for this case
        
        # Return loading screen first
        return loading_layout, create_notification(
            f"Started code generation for epic: {target_epic.title}", 
            "Epic Code Generation", 
            "info", 
            "code-slash"
        )
    
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error preparing epic code generation")
        return create_error_page(
            "Error Generating Code", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Process epic code generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
        Output("loading-status", "children", allow_duplicate=True),
        Output("loading-progress", "value", allow_duplicate=True),
    ],
    [
        Input("loading-status", "children"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def process_epic_generation(status, project_id, projects_data):
    """Process the epic code generation after showing loading screen"""
    if status != "Implementing all tasks in this epic... This may take a few moments.":
        return no_update, no_update, no_update, no_update, no_update
    
    try:
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, create_error_page(
                "Project Not Found", 
                "The project could not be found.",
                None
            ), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            ), no_update, no_update
        
        # Look for the epic ID from global context
        epic_id = None
        for button_id in ctx.inputs_list:
            if isinstance(button_id, dict) and button_id.get('type') == 'btn-generate-epic':
                epic_id = button_id.get('index')
                break
        
        if not epic_id:
            # Default to process all epics if we can't identify the specific one
            update_status = "Processing all epics..."
            return no_update, no_update, no_update, update_status, 30
        
        # Find target epic
        target_epic = None
        for epic in project.epics:
            if epic.id == epic_id:
                target_epic = epic
                break
        
        if not target_epic:
            return no_update, create_error_page(
                "Epic Not Found", 
                "The requested epic could not be found.",
                None
            ), create_notification(
                "Epic not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            ), no_update, no_update
        
        # Update status
        update_status = f"Processing epic: {target_epic.title}..."
        return no_update, no_update, no_update, update_status, 30
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error initializing epic code generation")
        return no_update, create_error_page(
            "Error Generating Code", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        ), no_update, no_update

# Execute epic code generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("loading-progress", "value"),
    ],
    [
        State("loading-status", "children"),
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def execute_epic_generation(progress, status, project_id, projects_data):
    """Execute epic code generation after loading status is updated"""
    if progress != 30 or not status or not status.startswith("Processing epic:"):
        return no_update, no_update, no_update
    
    try:
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, create_error_page(
                "Project Not Found", 
                "The project could not be found.",
                None
            ), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Extract epic name from status
        epic_title = status.replace("Processing epic: ", "").replace("...", "").strip()
        
        # Find target epic
        target_epic = None
        for epic in project.epics:
            if epic.title == epic_title:
                target_epic = epic
                break
        
        if not target_epic:
            return no_update, create_error_page(
                "Epic Not Found", 
                "The requested epic could not be found.",
                None
            ), create_notification(
                "Epic not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Process all tasks in this epic
        success_count = 0
        failed_count = 0
        
        for task in target_epic.tasks:
            if task.status != TaskStatus.COMPLETED:
                try:
                    updated_task = code_generator_service.generate_code_for_task(project, target_epic, task)
                    # Update task in project
                    for i, epic in enumerate(project.epics):
                        if epic.id == target_epic.id:
                            for j, t in enumerate(epic.tasks):
                                if t.id == task.id:
                                    project.epics[i].tasks[j] = updated_task
                                    break
                    
                    if updated_task.status == TaskStatus.COMPLETED:
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"Error generating code for task {task.title}: {str(e)}")
                    failed_count += 1
        
        # Save updated project
        projects.add_project(project)
        save_projects(projects)
        
        # Convert to dictionary for storage
        projects_dict = json.loads(projects.json())
        
        # Return to project detail view
        project_dict = projects_dict.get("projects", {}).get(project_id)
        return projects_dict, create_project_detail_layout(project_id, project_dict), create_notification(
            f"Epic code generation complete. Success: {success_count}, Failed: {failed_count}", 
            f"Epic: {target_epic.title}", 
            "success" if failed_count == 0 else "warning", 
            "check-circle" if failed_count == 0 else "exclamation-triangle"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error generating epic code")
        return no_update, create_error_page(
            "Error Generating Code", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Generate all code for a project
@callback(
    [
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("btn-generate-all", "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def generate_all_code(n_clicks, project_id, projects_data):
    """Generate code for all tasks in the project"""
    if not n_clicks:
        return no_update, no_update
    
    try:
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return create_error_page(
                "Project Not Found", 
                "The project could not be found.",
                None
            ), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Generate loading screen
        loading_layout = create_loading_screen(
            f"Generating All Code for Project: {project.name}",
            "Preparing to generate all code... This may take several minutes."
        )
        
        # Return loading screen first
        return loading_layout, create_notification(
            f"Started code generation for entire project: {project.name}", 
            "Project Code Generation", 
            "info", 
            "code-slash"
        )
        
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error initiating full project code generation")
        return create_error_page(
            "Error Generating Code", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Process full project code generation
@callback(
    [
        Output("loading-status", "children", allow_duplicate=True),
        Output("loading-progress", "value", allow_duplicate=True),
    ],
    [
        Input("loading-status", "children"),
    ],
    [
        State("loading-progress", "value"),
        State("current-project-id", "data"),
    ],
    prevent_initial_call=True
)
def update_generation_progress(status, progress, project_id):
    """Update the loading status during code generation"""
    if status != "Preparing to generate all code... This may take several minutes.":
        return no_update, no_update
    
    # Start the generation process
    return "Generating code for all epics and tasks...", 50

# Execute full project generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("loading-progress", "value"),
    ],
    [
        State("loading-status", "children"),
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def execute_full_generation(progress, status, project_id, projects_data):
    """Execute the full project code generation after loading screen appears"""
    if progress != 50 or status != "Generating code for all epics and tasks...":
        return no_update, no_update, no_update
    
    try:
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, create_error_page(
                "Project Not Found", 
                "The project could not be found.",
                None
            ), create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        # Generate code for all tasks
        try:
            updated_project = code_generator_service.generate_code_for_project(project)
            
            # Calculate generation statistics
            total_tasks = 0
            completed_tasks = 0
            failed_tasks = 0
            
            for epic in updated_project.epics:
                for task in epic.tasks:
                    total_tasks += 1
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks += 1
                    elif task.status == TaskStatus.FAILED:
                        failed_tasks += 1
            
            # Save updated project
            projects.add_project(updated_project)
            save_projects(projects)
            
            # Convert to dictionary for storage
            projects_dict = json.loads(projects.json())
            
            # Return to project detail view
            project_dict = projects_dict.get("projects", {}).get(project_id)
            return projects_dict, create_project_detail_layout(project_id, project_dict), create_notification(
                f"Project code generation complete. Success: {completed_tasks}, Failed: {failed_tasks}, Total: {total_tasks}", 
                "Code Generation Complete", 
                "success" if failed_tasks == 0 else "warning", 
                "check-circle" if failed_tasks == 0 else "exclamation-triangle"
            )
            
        except Exception as e:
            error_msg, error_details = handle_error(e, "Error during code generation")
            return no_update, create_error_page(
                "Error Generating Code", 
                error_msg,
                error_details
            ), create_notification(
                error_msg, 
                "Error", 
                "danger", 
                "exclamation-triangle"
            )
            
    except Exception as e:
        error_msg, error_details = handle_error(e, "Error generating project code")
        return no_update, create_error_page(
            "Error Generating Code", 
            error_msg,
            error_details
        ), create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Open output directory
@callback(
    [
        Output("btn-open-directory", "disabled"),
        Output("toast-message", "data", allow_duplicate=True),
    ],
    [
        Input("btn-open-directory", "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def open_output_directory(n_clicks, project_id, projects_data):
    """Open the output directory in file explorer"""
    if not n_clicks:
        return no_update, no_update
    
    try:
        project_data = projects_data.get("projects", {}).get(project_id)
        if not project_data:
            return no_update, create_notification(
                "Project not found", 
                "Error", 
                "warning", 
                "exclamation-triangle"
            )
        
        output_dir = project_data.get("output_directory")
        if not output_dir or not os.path.exists(output_dir):
            return no_update, create_notification(
                "Output directory not found or doesn't exist", 
                "Error", 
                "warning", 
                "folder-x"
            )
        
        try:
            # Windows-specific directory opening
            subprocess.Popen(f'explorer "{output_dir}"')
            return no_update, create_notification(
                f"Opened directory: {output_dir}", 
                "Directory Opened", 
                "info", 
                "folder2-open"
            )
        except Exception as e:
            return no_update, create_notification(
                f"Error opening directory: {str(e)}", 
                "Error", 
                "danger", 
                "folder-x"
            )
        
    except Exception as e:
        error_msg, _ = handle_error(e, "Error opening directory")
        return no_update, create_notification(
            error_msg, 
            "Error", 
            "danger", 
            "exclamation-triangle"
        )

# Toggle error details
@callback(
    [
        Output("error-details-collapse", "is_open"),
        Output("show-details-text", "children"),
    ],
    [Input("toggle-error-details", "n_clicks")],
    [State("error-details-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_error_details(n_clicks, is_open):
    """Toggle the visibility of error details"""
    if not n_clicks:
        return no_update, no_update
    
    if is_open:
        return False, [html.I(className="bi bi-info-circle me-2"), "Show Technical Details"]
    else:
        return True, [html.I(className="bi bi-info-circle-fill me-2"), "Hide Technical Details"]

# Copy code to clipboard - JS callback implementation
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            const code = document.querySelector('#code-preview-modal pre code');
            if (code) {
                const text = code.innerText;
                navigator.clipboard.writeText(text).then(
                    function() {
                        // Success
                        return true;
                    }, 
                    function() {
                        // Failure
                        return false;
                    }
                );
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("copy-code-button", "outline"),
    Input("copy-code-button", "n_clicks"),
    prevent_initial_call=True
)

# Navbar toggler
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_navbar_collapse(n_clicks, is_open):
    """Toggle navbar collapse on mobile"""
    if n_clicks:
        return not is_open
    return is_open

# Directory browser (just for show - would need backend integration)
@callback(
    Output("output-directory", "value"),
    Input("btn-browse-directory", "n_clicks"),
    State("output-directory", "value"),
    prevent_initial_call=True
)
def browse_directory(n_clicks, current_value):
    """Mock directory browser (would need actual system integration)"""
    if n_clicks:
        # For Windows, return a default projects folder in user directory
        # In a real implementation, this would open a directory picker
        return os.path.join(os.path.expanduser("~"), "Documents", "LLM-Projects")
    return current_value

# Search and filter projects (partial implementation)
@callback(
    Output("projects-store", "data", allow_duplicate=True),
    [
        Input("project-search", "value"),
        Input("sort-by-name", "n_clicks"),
        Input("sort-by-progress", "n_clicks"),
        Input("sort-by-date", "n_clicks"),
    ],
    [State("projects-store", "data")],
    prevent_initial_call=True
)
def filter_and_sort_projects(search_term, sort_name, sort_progress, sort_date, projects_data):
    """Filter and sort projects based on search term and sort options"""
    # This is a placeholder that would normally filter and sort the projects
    # In a real implementation, this would maintain the same data but change the order
    return projects_data  # No actual filtering in this demo