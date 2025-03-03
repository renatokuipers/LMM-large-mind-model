import os
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from config import get_config
from core.llm_manager import LLMManager, LLMConfig
from core.project_manager import ProjectManager
from planning.models import ProjectPlan, Epic, Task, TaskStatus, ComponentType
from planning.task_planner import TaskPlanner, EpicGenerationRequest
from planning.task_reviewer import TaskReviewer
from planning.task_executor import TaskExecutionManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Initialize LLM manager
llm_config = LLMConfig(
    api_base_url=config.llm.api_base_url,
    model_name=config.llm.model_name,
    api_key=config.llm.api_key,
    max_context_length=config.llm.max_context_length,
    temperature=config.llm.temperature,
    max_tokens=config.llm.max_tokens,
    timeout_seconds=config.llm.timeout_seconds
)
llm_manager = LLMManager(config=llm_config)

# Initialize project manager
output_dir = Path(config.app.output_dir)
project_manager = ProjectManager(llm_manager=llm_manager, output_base_dir=output_dir)

# Initialize planning components
task_planner = TaskPlanner(llm_manager=llm_manager)
task_reviewer = TaskReviewer(llm_manager=llm_manager)
task_executor = TaskExecutionManager(
    project_manager=project_manager,
    llm_manager=llm_manager,
    output_dir=output_dir
)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("LLM-powered Code Generator", className="mt-4 mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Project Specification"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Project Name", html_for="project-name"),
                                dbc.Input(
                                    type="text",
                                    id="project-name",
                                    placeholder="Enter project name",
                                    value=""
                                )
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Project Description", html_for="project-description"),
                                dbc.Textarea(
                                    id="project-description",
                                    placeholder="Describe your project in detail...",
                                    style={"height": "200px"},
                                    value=""
                                )
                            ])
                        ], className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Specific Requirements (one per line)", html_for="project-requirements"),
                                dbc.Textarea(
                                    id="project-requirements",
                                    placeholder="List specific requirements...",
                                    style={"height": "100px"},
                                    value=""
                                )
                            ])
                        ], className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Generate Project Plan",
                                    id="generate-plan-btn",
                                    color="primary",
                                    className="mt-3"
                                )
                            ])
                        ])
                    ])
                ])
            ], className="mb-4")
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Project Progress"),
                dbc.CardBody([
                    html.Div([
                        html.H5("Current Status:"),
                        html.P(id="project-status", children="Not started")
                    ]),
                    html.Div([
                        html.H5("Progress:"),
                        dcc.Graph(id="progress-graph", config={"displayModeBar": False})
                    ]),
                    html.Div([
                        dbc.Button(
                            "Review Plan",
                            id="review-plan-btn",
                            color="info",
                            className="me-2",
                            disabled=True
                        ),
                        dbc.Button(
                            "Execute Plan",
                            id="execute-plan-btn",
                            color="success",
                            disabled=True
                        )
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Generation Log"),
                dbc.CardBody([
                    dbc.Spinner(
                        html.Pre(
                            id="generation-log",
                            style={
                                "height": "200px",
                                "overflow-y": "auto",
                                "white-space": "pre-wrap"
                            }
                        )
                    )
                ])
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H3("Project Plan", className="mt-4", id="plan-heading"),
            html.Div(id="project-plan-view"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H3("Generated Components", className="mt-4", id="components-heading"),
            html.Div(id="component-list")
        ])
    ]),
    
    # Store components for sharing data between callbacks
    dcc.Store(id="project-plan-store"),
    dcc.Store(id="plan-review-store"),
    dcc.Store(id="generated-components-store"),
    dcc.Interval(id="progress-interval", interval=1000, disabled=True)
], fluid=True)


@app.callback(
    [
        Output("project-status", "children"),
        Output("generation-log", "children"),
        Output("project-plan-store", "data"),
        Output("review-plan-btn", "disabled"),
        Output("progress-graph", "figure")
    ],
    [Input("generate-plan-btn", "n_clicks")],
    [
        State("project-name", "value"),
        State("project-description", "value"),
        State("project-requirements", "value")
    ],
    prevent_initial_call=True
)
async def generate_project_plan(n_clicks, project_name, project_description, requirements):
    """Generate a project plan based on user inputs."""
    if not n_clicks or not project_name or not project_description:
        return "Not started", "", None, True, create_progress_figure(0, 0, 0)
    
    # Parse requirements
    req_list = [req.strip() for req in requirements.split("\n") if req.strip()]
    
    try:
        # Generate plan
        log_output = "Generating project plan...\n"
        request = EpicGenerationRequest(
            project_name=project_name,
            project_description=project_description,
            specific_requirements=req_list
        )
        
        plan = await task_planner.generate_project_plan(request)
        
        # Generate log output
        log_output += f"Plan generated with {len(plan.epics)} epics and {len(plan.all_tasks)} tasks\n"
        for i, epic in enumerate(plan.epics, 1):
            log_output += f"Epic {i}: {epic.title} ({len(epic.tasks)} tasks)\n"
        
        # Create initial progress graph
        figure = create_progress_figure(
            planned=len(plan.all_tasks),
            in_progress=0,
            completed=0
        )
        
        return (
            "Plan generated",
            log_output,
            plan.json(),
            False,  # Enable review button
            figure
        )
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        return (
            f"Error: {str(e)}",
            f"Error generating plan: {str(e)}",
            None,
            True,
            create_progress_figure(0, 0, 0)
        )


@app.callback(
    Output("project-plan-view", "children"),
    [Input("project-plan-store", "data")],
    prevent_initial_call=True
)
def display_project_plan(plan_json):
    """Display the project plan in a structured format."""
    if not plan_json:
        return "No plan generated yet."
    
    try:
        # Parse the plan
        plan = ProjectPlan.parse_raw(plan_json)
        
        # Create plan display
        plan_cards = []
        
        for i, epic in enumerate(plan.epics, 1):
            # Create epic card
            epic_header = f"Epic {i}: {epic.title}"
            if epic.status != TaskStatus.PLANNED:
                epic_header += f" ({epic.status.value})"
            
            epic_card = dbc.Card([
                dbc.CardHeader(epic_header),
                dbc.CardBody([
                    html.P(epic.description),
                    html.P(f"Complexity: {epic.estimated_complexity} | Priority: {epic.priority}"),
                    
                    # Tasks within this epic
                    html.H6(f"Tasks ({len(epic.tasks)}):"),
                    dbc.ListGroup([
                        create_task_item(task, j)
                        for j, task in enumerate(epic.tasks, 1)
                    ], flush=True)
                ])
            ], className="mb-3")
            
            plan_cards.append(epic_card)
        
        return plan_cards
    except Exception as e:
        logger.error(f"Error displaying plan: {str(e)}")
        return f"Error displaying plan: {str(e)}"


def create_task_item(task, index):
    """Create a ListGroupItem for a task."""
    # Determine color based on status
    if task.status == TaskStatus.COMPLETED:
        color = "success"
    elif task.status == TaskStatus.IN_PROGRESS:
        color = "info"
    elif task.status == TaskStatus.FAILED:
        color = "danger"
    else:
        color = None
    
    # Create the item
    return dbc.ListGroupItem([
        html.Div([
            html.Strong(f"Task {index}: {task.title}"),
            html.Span(f" ({task.component_type.value})", className="text-muted")
        ]),
        html.Small(f"Module: {task.module_path}"),
        html.P(task.description, className="mb-1"),
        html.Small([
            f"Complexity: {task.estimated_complexity} | ",
            f"Priority: {task.priority} | ",
            f"Status: {task.status.value}"
        ])
    ], color=color)


@app.callback(
    [
        Output("project-status", "children", allow_duplicate=True),
        Output("generation-log", "children", allow_duplicate=True),
        Output("plan-review-store", "data"),
        Output("execute-plan-btn", "disabled")
    ],
    [Input("review-plan-btn", "n_clicks")],
    [State("project-plan-store", "data"), State("generation-log", "children")],
    prevent_initial_call=True
)
async def review_project_plan(n_clicks, plan_json, current_log):
    """Review the generated project plan."""
    if not n_clicks or not plan_json:
        return "Plan generated", current_log, None, True
    
    try:
        # Parse the plan
        plan = ProjectPlan.parse_raw(plan_json)
        
        # Review the plan
        log_output = current_log + "\nReviewing project plan...\n"
        review = await task_reviewer.review_plan(plan)
        
        # Generate log output
        log_output += f"Plan review completed: {review.overall_assessment}\n"
        
        if review.issues:
            log_output += "\nIssues found:\n"
            for issue in review.issues:
                log_output += f"- {issue}\n"
        
        if review.suggestions:
            log_output += "\nSuggestions:\n"
            for suggestion in review.suggestions:
                log_output += f"- {suggestion}\n"
        
        if review.missing_components:
            log_output += "\nMissing components:\n"
            for component in review.missing_components:
                log_output += f"- {component}\n"
        
        # Determine if plan is approved
        status = "Plan reviewed - " + ("Approved" if review.is_approved else "Needs improvements")
        
        return (
            status,
            log_output,
            review.json(),
            not review.is_approved  # Enable execute button if approved
        )
    except Exception as e:
        logger.error(f"Error reviewing plan: {str(e)}")
        return (
            f"Error: {str(e)}",
            current_log + f"\nError reviewing plan: {str(e)}",
            None,
            True
        )


@app.callback(
    [
        Output("project-status", "children", allow_duplicate=True),
        Output("generation-log", "children", allow_duplicate=True),
        Output("progress-interval", "disabled"),
        Output("execute-plan-btn", "disabled", allow_duplicate=True)
    ],
    [Input("execute-plan-btn", "n_clicks")],
    [State("project-plan-store", "data"), State("generation-log", "children")],
    prevent_initial_call=True
)
async def start_plan_execution(n_clicks, plan_json, current_log):
    """Start executing the project plan."""
    if not n_clicks or not plan_json:
        return "Plan reviewed", current_log, True, False
    
    try:
        # Parse the plan
        plan = ProjectPlan.parse_raw(plan_json)
        
        # Start execution in a background task
        asyncio.create_task(execute_project_plan(plan))
        
        return (
            "Executing plan",
            current_log + "\nStarting plan execution...\n",
            False,  # Enable progress interval
            True    # Disable execute button
        )
    except Exception as e:
        logger.error(f"Error starting plan execution: {str(e)}")
        return (
            f"Error: {str(e)}",
            current_log + f"\nError starting plan execution: {str(e)}",
            True,
            False
        )


async def execute_project_plan(plan):
    """Execute the project plan in the background."""
    try:
        # Execute the plan
        generated_components = await task_executor.execute_plan(plan)
        
        # Store generated components for display
        app.generated_components = generated_components
    except Exception as e:
        logger.error(f"Error executing plan: {str(e)}")
        app.generated_components = {}


@app.callback(
    [
        Output("progress-graph", "figure", allow_duplicate=True),
        Output("generation-log", "children", allow_duplicate=True),
        Output("project-status", "children", allow_duplicate=True),
        Output("progress-interval", "disabled", allow_duplicate=True),
        Output("component-list", "children"),
        Output("project-plan-view", "children", allow_duplicate=True)
    ],
    [Input("progress-interval", "n_intervals")],
    [
        State("project-plan-store", "data"),
        State("generation-log", "children")
    ],
    prevent_initial_call=True
)
async def update_execution_progress(n_intervals, plan_json, current_log):
    """Update the execution progress display."""
    if not n_intervals or not plan_json:
        return (
            create_progress_figure(0, 0, 0),
            current_log, 
            "Executing plan", 
            False, 
            [],
            dash.no_update
        )
    
    try:
        # Parse the plan to get the current state
        plan = ProjectPlan.parse_raw(plan_json)
        
        # Get current progress
        completed = sum(1 for task in plan.all_tasks if task.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for task in plan.all_tasks if task.status == TaskStatus.IN_PROGRESS)
        failed = sum(1 for task in plan.all_tasks if task.status == TaskStatus.FAILED)
        planned = len(plan.all_tasks) - completed - in_progress - failed
        
        # Update log with newly completed tasks
        new_log = current_log
        component_cards = []
        
        # Check for completed tasks since last update
        for task in plan.all_tasks:
            if task.status == TaskStatus.COMPLETED and f"Completed: {task.title}" not in new_log:
                new_log += f"Completed: {task.title} ({task.component_type})\n"
        
        # Add generated components if available
        if hasattr(app, 'generated_components') and app.generated_components:
            for component_name, file_path in app.generated_components.items():
                # Read component file
                try:
                    with open(file_path, 'r') as f:
                        code = f.read()
                except Exception:
                    code = "Error reading file"
                
                # Create component card
                card = dbc.Card([
                    dbc.CardHeader(component_name),
                    dbc.CardBody([
                        html.P(f"File: {file_path}"),
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody(html.Pre(code)), className="mt-2"),
                            id=f"collapse-{hash(component_name)}",
                            is_open=False
                        ),
                        dbc.Button(
                            "Show/Hide Code",
                            id=f"btn-{hash(component_name)}",
                            className="mt-2",
                            color="primary",
                            size="sm"
                        )
                    ])
                ], className="mb-3")
                
                component_cards.append(card)
        
        # Update progress figure
        figure = create_progress_figure(planned, in_progress, completed, failed)
        
        # Update status text
        if completed + failed == len(plan.all_tasks):
            status = "Plan execution complete!"
            new_log += "\nPlan execution complete!\n"
            interval_disabled = True
        else:
            status = f"Executing plan ({completed}/{len(plan.all_tasks)} tasks completed)"
            interval_disabled = False
        
        # Refresh plan view to show updated task statuses
        plan_view = display_project_plan(plan_json)
        
        return figure, new_log, status, interval_disabled, component_cards, plan_view
        
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")
        return (
            create_progress_figure(0, 0, 0),
            current_log + f"\nError updating progress: {str(e)}",
            "Error updating progress",
            True,
            [],
            dash.no_update
        )


def create_progress_figure(planned, in_progress, completed, failed=0):
    """Create a progress figure."""
    labels = ['Planned', 'In Progress', 'Completed', 'Failed']
    values = [planned, in_progress, completed, failed]
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
    
    # Filter out categories with zero values
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    
    for i, value in enumerate(values):
        if value > 0:
            filtered_labels.append(labels[i])
            filtered_values.append(value)
            filtered_colors.append(colors[i])
    
    # If no data, add empty state
    if not filtered_values:
        filtered_labels = ['No tasks']
        filtered_values = [1]
        filtered_colors = ['lightgray']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=filtered_labels,
            values=filtered_values,
            hole=.3,
            marker_colors=filtered_colors
        )
    ])
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        showlegend=True
    )
    
    return fig


if __name__ == "__main__":
    app.run_server(debug=config.app.debug, port=8050)