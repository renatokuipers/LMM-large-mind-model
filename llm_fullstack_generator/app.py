# app.py
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import logging
from datetime import datetime
import os

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

# Initialize the application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Create the layout
app.layout = create_layout()

# Initialize orchestrator
orchestrator = Orchestrator(output_dir="./generated_projects")

@callback(
    Output("project-output", "children"),
    Input("generate-button", "n_clicks"),
    State("project-name", "value"),
    State("project-description", "value"),
    State("language-dropdown", "value"),
    State("framework-input", "value"),
    State("database-dropdown", "value"),
    State("frontend-toggle", "value"),
    State("frontend-framework", "value"),
    prevent_initial_call=True
)
def generate_project(n_clicks, name, description, language, framework, database, include_frontend, frontend_framework):
    """Generate a project based on user inputs"""
    if not n_clicks or not name or not description:
        return html.Div("Please provide a project name and description.")
    
    try:
        # Create project config
        config = ProjectConfig(
            language=language or "python",
            framework=framework,
            database=database,
            include_frontend=bool(include_frontend),
            frontend_framework=frontend_framework if include_frontend else None
        )
        
        # Initialize project
        logger.info(f"Initializing project: {name}")
        project_context = orchestrator.initialize_project(name, description, config)
        
        # Generate code
        logger.info("Starting code generation")
        success = orchestrator.run_full_generation()
        
        # Prepare output
        output_components = []
        
        # Project info
        output_components.append(html.H3(f"Project: {name}"))
        output_components.append(html.P(f"Status: {'Completed' if success else 'Completed with some issues'}"))
        output_components.append(html.P(f"Output directory: {project_context.output_dir}"))
        
        # EPICs and tasks
        output_components.append(html.H4("Project Plan"))
        for epic in project_context.epics:
            epic_card = dbc.Card([
                dbc.CardHeader(html.H5(f"EPIC: {epic.title}")),
                dbc.CardBody([
                    html.P(epic.description),
                    html.H6("Tasks:"),
                    html.Ul([
                        html.Li([
                            html.Strong(f"{task.title} "), 
                            html.Span(f"({task.status})")
                        ])
                        for task in epic.tasks
                    ])
                ])
            ], className="mb-3")
            output_components.append(epic_card)
        
        # Add open folder button
        output_components.append(
            dbc.Button(
                "Open Project Folder",
                id="open-folder-button",
                color="primary",
                className="mt-3",
                href=f"file://{os.path.abspath(project_context.output_dir)}",
                external_link=True,
                target="_blank"
            )
        )
        
        return html.Div(output_components)
        
    except Exception as e:
        logger.error(f"Error generating project: {str(e)}")
        return html.Div([
            html.H4("Error Generating Project", className="text-danger"),
            html.P(str(e))
        ])

# UI Layouts
from ui.layouts import create_layout
from ui.callbacks import register_callbacks

# Register all callbacks
register_callbacks(app, orchestrator)

if __name__ == "__main__":
    logger.info("Starting LLM Fullstack Generator")
    app.run_server(debug=True, port=8050)