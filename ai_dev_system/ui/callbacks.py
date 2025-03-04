import dash
from dash import callback, Input, Output, State, html, dcc, no_update, ctx
import dash_bootstrap_components as dbc
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Import app from app.py
from ui.app import app, create_projects_layout, create_new_project_layout, create_project_detail_layout, create_task_detail_modal

# Import models and services
from models.project_model import Project, Epic, Task, ProjectStore
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
def load_projects():
    """Load projects from file"""
    try:
        return ProjectStore.load_from_file(PROJECTS_FILE)
    except Exception as e:
        print(f"Error loading projects: {str(e)}")
        return ProjectStore()

def save_projects(projects: ProjectStore):
    """Save projects to file"""
    try:
        projects.save_to_file(PROJECTS_FILE)
    except Exception as e:
        print(f"Error saving projects: {str(e)}")

# Navigation callbacks
@callback(
    Output("page-content", "children"),
    [
        Input("nav-projects", "n_clicks"),
        Input("nav-create", "n_clicks"),
        Input("btn-create-project", "n_clicks"),
        Input("btn-back-to-projects", "n_clicks"),
        Input("btn-cancel-project", "n_clicks"),
        Input({"type": "btn-view-project", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("projects-store", "data"),
        State("current-project-id", "data"),
    ],
    prevent_initial_call=True
)
def navigate(nav_projects, nav_create, btn_create, btn_back, btn_cancel, btn_view, 
            projects_data, current_project_id):
    """Handle navigation between pages"""
    triggered_id = ctx.triggered_id
    
    # Load projects data
    projects = load_projects()
    projects_dict = json.loads(projects.json()) if projects else {"projects": {}}
    
    # Navigate to projects list
    if triggered_id == "nav-projects" or triggered_id == "btn-back-to-projects" or triggered_id == "btn-cancel-project":
        return create_projects_layout(projects_dict)
    
    # Navigate to create new project form
    elif triggered_id == "nav-create" or triggered_id == "btn-create-project":
        return create_new_project_layout()
    
    # Navigate to project detail
    elif isinstance(triggered_id, dict) and triggered_id.get("type") == "btn-view-project":
        project_id = triggered_id.get("index")
        project_data = projects_dict.get("projects", {}).get(project_id)
        if project_data:
            return create_project_detail_layout(project_id, project_data)
    
    # Default to projects list
    return create_projects_layout(projects_dict)

# Project creation callback
@callback(
    [
        Output("projects-store", "data"),
        Output("current-project-id", "data"),
        Output("page-content", "children", allow_duplicate=True),
    ],
    [
        Input("btn-submit-project", "n_clicks"),
    ],
    [
        State("project-name", "value"),
        State("project-description", "value"),
        State("output-directory", "value"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def create_project(n_clicks, name, description, output_directory, existing_projects_data):
    """Create a new project and generate its plan"""
    if not n_clicks or not name or not description or not output_directory:
        return no_update, no_update, no_update
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Create project object
    project = Project(
        name=name,
        description=description,
        output_directory=output_directory
    )
    
    # Generate project plan
    loading_layout = dbc.Container([
        html.H2(f"Creating Project: {name}", className="mb-4"),
        dbc.Spinner(size="lg", color="primary", type="grow"),
        html.Div(id="planning-status", className="mt-4"),
    ])
    
    # Update UI immediately with loading spinner
    return no_update, project.id, loading_layout

# Project planning process
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
    ],
    [
        Input("current-project-id", "data"),
    ],
    [
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def plan_project(project_id, existing_projects_data):
    """Generate plan for the newly created project"""
    if not project_id:
        return no_update, no_update
    
    # Load existing projects
    projects = load_projects()
    project = projects.get_project(project_id)
    
    if not project:
        # Project not found in store, check if it's a newly created project
        if existing_projects_data and existing_projects_data.get("projects", {}).get(project_id):
            # Project exists in the store but not in our ProjectStore object
            project_data = existing_projects_data.get("projects", {}).get(project_id)
            project = Project.parse_obj(project_data)
        else:
            # This should not happen if create_project callback worked correctly
            return no_update, dbc.Alert("Error: Project not found", color="danger")
    
    try:
        # Generate project plan
        project = planner_service.create_project_plan(project)
        
        # Save updated project
        projects.add_project(project)
        save_projects(projects)
        
        # Convert to dictionary for storage
        projects_dict = json.loads(projects.json())
        
        # Return to project detail view
        project_dict = projects_dict.get("projects", {}).get(project_id)
        return projects_dict, create_project_detail_layout(project_id, project_dict)
        
    except Exception as e:
        print(f"Error planning project: {str(e)}")
        return no_update, dbc.Alert(f"Error planning project: {str(e)}", color="danger")

# Task detail modal
@callback(
    Output("task-detail-modal", "is_open"),
    [
        Input({"type": "btn-task-details", "epic": dash.ALL, "task": dash.ALL}, "n_clicks"),
        Input("close-task-modal", "n_clicks"),
    ],
    prevent_initial_call=True
)
def toggle_task_modal(btn_clicks, close_click):
    """Open or close the task detail modal"""
    ctx_triggered = ctx.triggered_id
    
    if ctx_triggered == "close-task-modal":
        return False
    
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "btn-task-details":
        return True
    
    return no_update

# Populate task detail modal
@callback(
    Output("task-detail-modal", "children"),
    [
        Input({"type": "btn-task-details", "epic": dash.ALL, "task": dash.ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def show_task_details(btn_clicks, project_id, projects_data):
    """Show task details in modal"""
    ctx_triggered = ctx.triggered_id
    
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "btn-task-details":
        epic_id = ctx_triggered.get("epic")
        task_id = ctx_triggered.get("task")
        
        project_data = projects_data.get("projects", {}).get(project_id)
        
        return create_task_detail_modal(epic_id, task_id, project_data)
    
    return no_update

# Generate code for a specific task
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("task-detail-modal", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "btn-generate-task", "epic": dash.ALL, "task": dash.ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def generate_task_code(btn_clicks, project_id, projects_data):
    """Generate code for a specific task"""
    ctx_triggered = ctx.triggered_id
    
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "btn-generate-task":
        epic_id = ctx_triggered.get("epic")
        task_id = ctx_triggered.get("task")
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, dbc.Modal([
                dbc.ModalHeader("Error"),
                dbc.ModalBody("Project not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ml-auto")),
            ], id="task-detail-modal")
        
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
                dbc.ModalHeader("Error"),
                dbc.ModalBody("Epic or task not found"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ml-auto")),
            ], id="task-detail-modal")
        
        try:
            # Generate loading modal
            loading_modal = dbc.Modal([
                dbc.ModalHeader(f"Generating Code: {target_task.title}"),
                dbc.ModalBody([
                    dbc.Spinner(size="lg", color="primary", type="grow"),
                    html.P("This may take a few moments...", className="mt-3"),
                ]),
            ], id="task-detail-modal", is_open=True)
            
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
            return projects_dict, create_task_detail_modal(epic_id, task_id, projects_dict.get("projects", {}).get(project_id))
            
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return no_update, dbc.Modal([
                dbc.ModalHeader("Error"),
                dbc.ModalBody(f"Error generating code: {str(e)}"),
                dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ml-auto")),
            ], id="task-detail-modal")
    
    return no_update, no_update

# Generate code for an entire epic
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
    ],
    [
        Input({"type": "btn-generate-epic", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def generate_epic_code(btn_clicks, project_id, projects_data):
    """Generate code for all tasks in an epic"""
    ctx_triggered = ctx.triggered_id
    
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "btn-generate-epic":
        epic_id = ctx_triggered.get("index")
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, dbc.Alert("Project not found", color="danger")
        
        # Find epic
        target_epic = None
        for epic in project.epics:
            if epic.id == epic_id:
                target_epic = epic
                break
        
        if not target_epic:
            return no_update, dbc.Alert("Epic not found", color="danger")
        
        # Generate loading screen
        loading_layout = dbc.Container([
            html.H2(f"Generating Code for Epic: {target_epic.title}", className="mb-4"),
            dbc.Spinner(size="lg", color="primary", type="grow"),
            html.Div(id="generation-status", className="mt-4"),
        ])
        
        # Return loading screen first
        return no_update, loading_layout
    
    return no_update, no_update

# Process epic code generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
    ],
    [
        Input("generation-status", "children"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def process_epic_generation(status, project_id, projects_data):
    """Process the epic code generation after showing loading screen"""
    if status is None:
        # Only triggered by the loading screen being shown
        time.sleep(0.5)  # Small delay to ensure UI updates
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, dbc.Alert("Project not found", color="danger")
        
        try:
            # We can't directly know which epic to process here, so process all
            for epic in project.epics:
                for task in epic.tasks:
                    if task.status != "completed":
                        updated_task = code_generator_service.generate_code_for_task(project, epic, task)
                        # Update the task in the project
                        for i, e in enumerate(project.epics):
                            if e.id == epic.id:
                                for j, t in enumerate(e.tasks):
                                    if t.id == task.id:
                                        project.epics[i].tasks[j] = updated_task
                                        break
                                break
            
            # Save updated project
            projects.add_project(project)
            save_projects(projects)
            
            # Convert to dictionary for storage
            projects_dict = json.loads(projects.json())
            
            # Return to project detail view
            project_dict = projects_dict.get("projects", {}).get(project_id)
            return projects_dict, create_project_detail_layout(project_id, project_dict)
            
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return no_update, dbc.Alert(f"Error generating code: {str(e)}", color="danger")
    
    return no_update, no_update

# Generate all code for a project
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
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
    
    # Load project
    projects = load_projects()
    project = projects.get_project(project_id)
    
    if not project:
        return no_update, dbc.Alert("Project not found", color="danger")
    
    # Generate loading screen
    loading_layout = dbc.Container([
        html.H2(f"Generating All Code for Project: {project.name}", className="mb-4"),
        dbc.Spinner(size="lg", color="primary", type="grow"),
        html.Div(id="full-generation-status", className="mt-4"),
    ])
    
    # Return loading screen first
    return no_update, loading_layout

# Process full project code generation
@callback(
    [
        Output("projects-store", "data", allow_duplicate=True),
        Output("page-content", "children", allow_duplicate=True),
    ],
    [
        Input("full-generation-status", "children"),
    ],
    [
        State("current-project-id", "data"),
        State("projects-store", "data"),
    ],
    prevent_initial_call=True
)
def process_full_generation(status, project_id, projects_data):
    """Process the full project code generation after showing loading screen"""
    if status is None:
        # Only triggered by the loading screen being shown
        time.sleep(0.5)  # Small delay to ensure UI updates
        
        # Load project
        projects = load_projects()
        project = projects.get_project(project_id)
        
        if not project:
            return no_update, dbc.Alert("Project not found", color="danger")
        
        try:
            # Generate code for all tasks
            updated_project = code_generator_service.generate_code_for_project(project)
            
            # Save updated project
            projects.add_project(updated_project)
            save_projects(projects)
            
            # Convert to dictionary for storage
            projects_dict = json.loads(projects.json())
            
            # Return to project detail view
            project_dict = projects_dict.get("projects", {}).get(project_id)
            return projects_dict, create_project_detail_layout(project_id, project_dict)
            
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return no_update, dbc.Alert(f"Error generating code: {str(e)}", color="danger")
    
    return no_update, no_update

# Open output directory
@callback(
    Output("btn-open-directory", "disabled"),
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
        return no_update
    
    project_data = projects_data.get("projects", {}).get(project_id)
    if not project_data:
        return no_update
    
    output_dir = project_data.get("output_directory")
    if not output_dir or not os.path.exists(output_dir):
        return no_update
    
    try:
        # Open directory in file explorer
        os.startfile(output_dir)
    except Exception as e:
        print(f"Error opening directory: {str(e)}")
    
    return no_update