# app.py
import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import logging
from datetime import datetime
import os
import time
import threading
from flask import request

from core.schemas import ProjectConfig
from core.orchestrator import Orchestrator
from ui.layouts import create_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize the application with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],  # Changed from BOOTSTRAP to DARKLY
    suppress_callback_exceptions=True
)

# Create the layout
app.layout = create_layout()

# Initialize orchestrator
orchestrator = Orchestrator(output_dir="./generated_projects")

# Store for project generation status
generation_status = {
    "is_generating": False,
    "progress": 0,
    "current_task": "",
    "log_messages": [],
    "completed_tasks": [],
    "project_context": None,
    "success": False
}

def run_generation(name, description, config):
    """Run the generation process in a separate thread"""
    global generation_status
    
    try:
        # Update status
        generation_status["is_generating"] = True
        generation_status["progress"] = 5
        generation_status["current_task"] = "Initializing project..."
        generation_status["log_messages"].append(f"Starting project generation: {name}")
        
        # Initialize project
        logger.info(f"Initializing project: {name}")
        project_context = orchestrator.initialize_project(name, description, config)
        generation_status["project_context"] = project_context
        generation_status["progress"] = 20
        generation_status["log_messages"].append(f"Project initialization complete. Created {len(project_context.epics)} epics.")
        
        # Generate code
        logger.info("Starting code generation")
        generation_status["current_task"] = "Generating code..."
        
        # Get total task count for progress calculation
        total_tasks = sum(len(epic.tasks) for epic in project_context.epics)
        completed_tasks = 0
        
        # Process tasks one by one
        while True:
            next_task = orchestrator.get_next_task()
            if not next_task:
                break
                
            generation_status["current_task"] = f"Generating: {next_task.title}"
            generation_status["log_messages"].append(f"Working on task: {next_task.title}")
            
            success = orchestrator.execute_task(next_task)
            completed_tasks += 1
            generation_status["progress"] = 20 + int(70 * (completed_tasks / total_tasks))
            
            if success:
                generation_status["completed_tasks"].append(next_task.id)
                generation_status["log_messages"].append(f"✅ Completed: {next_task.title}")
            else:
                generation_status["log_messages"].append(f"❌ Failed: {next_task.title}")
        
        # Check if all tasks are completed
        all_completed = True
        for epic in project_context.epics:
            for task in epic.tasks:
                if task.status != "completed":
                    all_completed = False
                    break
        
        generation_status["success"] = all_completed
        generation_status["progress"] = 100
        generation_status["current_task"] = "Project generation complete!"
        generation_status["log_messages"].append(f"Project generation {'completed successfully' if all_completed else 'finished with some failures'}")
        
    except Exception as e:
        logger.error(f"Error in generation thread: {str(e)}")
        generation_status["log_messages"].append(f"Error: {str(e)}")
    finally:
        generation_status["is_generating"] = False


@callback(
    [
        Output("project-output", "children"),
        Output("generate-button", "disabled"),
        Output("generation-progress", "value"),
        Output("generation-progress", "label"),
        Output("generation-progress", "color"),
        Output("generation-log", "children")
    ],
    [
        Input("generate-button", "n_clicks"),
        Input("status-interval", "n_intervals")
    ],
    [
        State("project-name", "value"),
        State("project-description", "value"),
        State("language-dropdown", "value"),
        State("framework-input", "value"),
        State("database-dropdown", "value"),
        State("frontend-toggle", "value"),
        State("frontend-framework", "value")
    ],
    prevent_initial_call=True
)
def handle_generation(n_clicks, n_intervals, name, description, language, framework, 
                      database, include_frontend, frontend_framework):
    """Handle project generation and status updates"""
    global generation_status
    
    triggered = ctx.triggered_id
    
    # Default return values
    output_components = html.Div("Enter project details and click 'Generate Project'")
    generate_button_disabled = False
    progress_value = 0
    progress_label = ""
    progress_color = "primary"
    log_components = []
    
    # If Generate button was clicked
    if triggered == "generate-button" and n_clicks:
        if not name or not description:
            return [
                html.Div("Please provide a project name and description.", className="text-warning"),
                False, 0, "", "primary", []
            ]
            
        # Create project config
        config = ProjectConfig(
            language=language or "python",
            framework=framework,
            database=database,
            include_frontend=bool(include_frontend),
            frontend_framework=frontend_framework if include_frontend else None
        )
        
        # Reset status
        generation_status = {
            "is_generating": False,
            "progress": 0,
            "current_task": "",
            "log_messages": [],
            "completed_tasks": [],
            "project_context": None,
            "success": False
        }
        
        # Start generation in a separate thread
        thread = threading.Thread(target=run_generation, args=(name, description, config))
        thread.daemon = True
        thread.start()
        
        # Initial UI update
        return [
            html.Div([
                html.H3(f"Generating Project: {name}", className="text-primary"),
                html.P("Generation has started. Please wait...", className="text-info")
            ]),
            True, 0, "Starting...", "primary", []
        ]
    
    # Status interval updates
    if triggered == "status-interval":
        # If generation is in progress or just completed
        if generation_status["is_generating"] or generation_status["progress"] == 100:
            # Create log messages
            log_components = [
                html.Div(msg, className="mb-1 small") 
                for msg in generation_status["log_messages"]
            ]
            
            # Progress indicator
            progress_value = generation_status["progress"]
            progress_label = f"{progress_value}% - {generation_status['current_task']}"
            progress_color = "success" if generation_status["progress"] == 100 else "primary"
            
            # If complete, create final output
            if generation_status["progress"] == 100 and generation_status["project_context"]:
                project_context = generation_status["project_context"]
                
                # Project info
                output_components = [
                    html.H3(f"Project: {project_context.name}", className="text-success"),
                    html.P(f"Status: {'Completed' if generation_status['success'] else 'Completed with some issues'}", 
                           className="text-info" if generation_status["success"] else "text-warning"),
                    html.P(f"Output directory: {project_context.output_dir}", className="text-info mb-4")
                ]
                
                # EPICs and tasks
                output_components.append(html.H4("Project Plan", className="mt-3"))
                for epic in project_context.epics:
                    epic_card = dbc.Card([
                        dbc.CardHeader(html.H5(f"EPIC: {epic.title}", className="mb-0")),
                        dbc.CardBody([
                            html.P(epic.description),
                            html.H6("Tasks:", className="mt-3"),
                            html.Ul([
                                html.Li([
                                    html.Strong(f"{task.title} ", className="me-2"), 
                                    html.Span(f"({task.status})", 
                                             className="text-success" if task.status == "completed" else 
                                                      "text-danger" if task.status == "failed" else "text-warning")
                                ])
                                for task in epic.tasks
                            ], className="mb-0")
                        ])
                    ], className="mb-3 bg-dark border-secondary")
                    output_components.append(epic_card)
                
                # Add open folder button
                output_components.append(
                    dbc.Button(
                        [html.I(className="fas fa-folder-open me-2"), "Open Project Folder"],
                        id="open-folder-button",
                        color="primary",
                        className="mt-3",
                        href=f"file://{os.path.abspath(project_context.output_dir)}",
                        external_link=True,
                        target="_blank"
                    )
                )
                
                # Wrap in a div
                output_components = html.Div(output_components)
                
                # Enable the generate button again
                generate_button_disabled = False
            else:
                # Still generating
                output_components = html.Div([
                    html.H3("Project Generation In Progress", className="text-primary"),
                    html.P(f"Current task: {generation_status['current_task']}", className="text-info"),
                    html.P(f"Completed tasks: {len(generation_status['completed_tasks'])}", className="text-info")
                ])
                generate_button_disabled = True
                
            return [
                output_components,
                generate_button_disabled,
                progress_value,
                progress_label,
                progress_color,
                log_components
            ]
    
    # Default return
    return [output_components, generate_button_disabled, progress_value, progress_label, progress_color, log_components]


# UI Layouts
from ui.layouts import create_layout
from ui.callbacks import register_callbacks

# Register all callbacks
register_callbacks(app, orchestrator)

if __name__ == "__main__":
    logger.info("Starting LLM Fullstack Generator")
    app.run_server(debug=True, port=8050)