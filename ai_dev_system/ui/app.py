import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import json
from typing import Dict, List

# Initialize Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

# Projects list page
def create_projects_layout(projects_data=None):
    if not projects_data or not projects_data.get('projects'):
        return dbc.Container([
            html.H2("Your Projects", className="mb-4"),
            html.P("No projects found. Create your first project!"),
            dbc.Button("Create New Project", id="btn-create-project", color="primary"),
        ])
    
    # Rest of the function remains the same...
    projects = projects_data.get('projects', {})
    project_cards = []
    
    for project_id, project in projects.items():
        # Calculate completion percentage
        total_tasks = 0
        completed_tasks = 0
        for epic in project.get('epics', []):
            for task in epic.get('tasks', []):
                total_tasks += 1
                if task.get('status') == 'completed':
                    completed_tasks += 1
        
        completion = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Create card for each project
        card = dbc.Card([
            dbc.CardHeader(project.get('name', 'Unnamed Project')),
            dbc.CardBody([
                html.H5(project.get('name', 'Unnamed Project'), className="card-title"),
                html.P(project.get('description', 'No description'), className="card-text"),
                html.Div([
                    html.Span(f"Progress: {completion:.1f}%", className="me-2"),
                    dbc.Progress(value=completion, className="mb-3"),
                ]),
                html.P(f"Created: {project.get('created_at', 'Unknown')}", className="text-muted small"),
                dbc.Button(
                    "View Project", 
                    id={"type": "btn-view-project", "index": project_id},
                    color="primary", 
                    className="me-2"
                ),
            ]),
        ], className="mb-4")
        
        project_cards.append(card)
    
    return dbc.Container([
        html.H2("Your Projects", className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Button("Create New Project", id="btn-create-project", color="success"), width="auto"),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(card, md=4) for card in project_cards
        ]),
    ])

# Create new project form
def create_new_project_layout():
    return dbc.Container([
        html.H2("Create New Project", className="mb-4"),
        
        dbc.Form([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project Name", html_for="project-name"),
                    dbc.Input(type="text", id="project-name", placeholder="Enter project name"),
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project Description", html_for="project-description"),
                    dbc.Textarea(
                        id="project-description",
                        placeholder="Describe your project in detail...",
                        style={"height": "150px"}
                    ),
                ]),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Output Directory", html_for="output-directory"),
                    dbc.Input(
                        type="text",
                        id="output-directory",
                        placeholder="C:\\path\\to\\output\\directory"
                    ),
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Create Project", id="btn-submit-project", color="primary"),
                    dbc.Button("Cancel", id="btn-cancel-project", color="secondary", className="ms-2"),
                ]),
            ]),
        ]),
    ])

# Project detail view
def create_project_detail_layout(project_id, project_data):
    if not project_data:
        return html.Div("Project not found")
    
    # Extract epics and tasks
    epics = project_data.get('epics', [])
    
    # Create epic accordion items
    epic_items = []
    for i, epic in enumerate(epics):
        # Calculate epic progress
        total_tasks = len(epic.get('tasks', []))
        completed_tasks = sum(1 for task in epic.get('tasks', []) if task.get('status') == 'completed')
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Create tasks table
        tasks_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Task"),
                    html.Th("Status"),
                    html.Th("Actions"),
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(task.get('title', 'Unnamed Task')),
                    html.Td(
                        dbc.Badge(
                            task.get('status', 'planned').upper(),
                            color={"planned": "secondary", "in_progress": "primary", "completed": "success", "failed": "danger"}.get(task.get('status', 'planned'), "secondary"),
                            className="me-1"
                        )
                    ),
                    html.Td(
                        dbc.Button(
                            "Details", 
                            id={"type": "btn-task-details", "epic": epic.get('id'), "task": task.get('id')},
                            size="sm",
                            color="info"
                        )
                    ),
                ]) for task in epic.get('tasks', [])
            ]),
        ], bordered=True, hover=True, responsive=True, size="sm", dark=True)
        
        # Create epic card
        epic_item = dbc.AccordionItem(
            [
                html.Div([
                    html.P(epic.get('description', 'No description')),
                    html.Div([
                        html.Span(f"Progress: {progress:.1f}% ({completed_tasks}/{total_tasks} tasks)", className="me-2"),
                        dbc.Progress(value=progress, className="mb-3"),
                    ]),
                    tasks_table,
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Generate Code for Epic", 
                                id={"type": "btn-generate-epic", "index": epic.get('id')},
                                color="primary",
                                className="mt-3 me-2"
                            ),
                        ]),
                    ]),
                ]),
            ],
            title=f"Epic {i+1}: {epic.get('title', 'Unnamed Epic')}",
            item_id=epic.get('id', f"epic-{i}")
        )
        
        epic_items.append(epic_item)
    
    # Tech stack badges
    tech_stack = project_data.get('tech_stack', {})
    tech_badges = []
    
    if tech_stack.get('frontend'):
        for tech in tech_stack.get('frontend', []):
            tech_badges.append(dbc.Badge(f"Frontend: {tech}", color="info", className="me-1 mb-1"))
    
    if tech_stack.get('backend'):
        for tech in tech_stack.get('backend', []):
            tech_badges.append(dbc.Badge(f"Backend: {tech}", color="success", className="me-1 mb-1"))
    
    if tech_stack.get('database'):
        for tech in tech_stack.get('database', []):
            tech_badges.append(dbc.Badge(f"DB: {tech}", color="warning", className="me-1 mb-1"))
    
    # Calculate overall project progress
    total_tasks = 0
    completed_tasks = 0
    for epic in epics:
        for task in epic.get('tasks', []):
            total_tasks += 1
            if task.get('status') == 'completed':
                completed_tasks += 1
    
    overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2(project_data.get('name', 'Unnamed Project'), className="mb-3"),
                html.P(project_data.get('description', 'No description'), className="lead mb-3"),
                html.Div(tech_badges, className="mb-3"),
                html.Div([
                    html.Span(f"Overall Progress: {overall_progress:.1f}% ({completed_tasks}/{total_tasks} tasks)", className="me-2"),
                    dbc.Progress(value=overall_progress, className="mb-4"),
                ]),
            ]),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("Back to Projects", id="btn-back-to-projects", color="secondary", className="me-2"),
                dbc.Button("Generate All Code", id="btn-generate-all", color="success", className="me-2"),
                dbc.Button("Open Output Directory", id="btn-open-directory", color="info"),
            ], className="mb-4"),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Project Plan", className="mb-3"),
                dbc.Accordion(epic_items, start_collapsed=True, always_open=False),
            ]),
        ]),
    ])

# Task detail modal
def create_task_detail_modal(epic_id, task_id, project_data):
    # Find epic and task
    epic = None
    task = None
    
    for e in project_data.get('epics', []):
        if e.get('id') == epic_id:
            epic = e
            for t in e.get('tasks', []):
                if t.get('id') == task_id:
                    task = t
                    break
            break
    
    if not epic or not task:
        return dbc.Modal([
            dbc.ModalHeader("Error"),
            dbc.ModalBody("Task not found"),
            dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ml-auto")),
        ], id="task-detail-modal")
    
    # Create code items table
    code_items = task.get('code_items', [])
    code_items_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Type"),
                html.Th("Name"),
                html.Th("File"),
                html.Th("Status"),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(item.get('type', 'Unknown')),
                html.Td(item.get('name', 'Unnamed')),
                html.Td(item.get('file_path', 'Not specified')),
                html.Td(
                    dbc.Badge(
                        "Implemented" if item.get('implemented', False) else "Pending",
                        color="success" if item.get('implemented', False) else "secondary",
                    )
                ),
            ]) for item in code_items
        ]),
    ], bordered=True, hover=True, responsive=True, size="sm", dark=True)
    
    return dbc.Modal([
        dbc.ModalHeader(f"Task: {task.get('title', 'Unnamed Task')}"),
        dbc.ModalBody([
            html.H6("Description"),
            html.P(task.get('description', 'No description')),
            
            html.H6("Status"),
            dbc.Badge(
                task.get('status', 'planned').upper(),
                color={"planned": "secondary", "in_progress": "primary", "completed": "success", "failed": "danger"}.get(task.get('status', 'planned'), "secondary"),
                className="mb-3"
            ),
            
            html.H6("Code Items"),
            code_items_table if code_items else html.P("No code items defined for this task."),
            
            html.H6("Created"),
            html.P(task.get('created_at', 'Unknown')),
            
            html.H6("Last Updated"),
            html.P(task.get('updated_at', 'Unknown')),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Generate Code", 
                id={"type": "btn-generate-task", "epic": epic_id, "task": task_id},
                color="primary",
                className="me-2",
                disabled=task.get('status') == 'completed'
            ),
            dbc.Button("Close", id="close-task-modal", className="ml-auto"),
        ]),
    ], id="task-detail-modal", size="lg")
    
# App layout
app.layout = html.Div([
    dcc.Store(id='projects-store', storage_type='local'),
    dcc.Store(id='current-project-id', storage_type='local'),
    dcc.Store(id='current-epic-id', storage_type='local'),
    
    # Navigation bar
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("LLM Full-Stack Developer", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Projects", href="#", id="nav-projects")),
                dbc.NavItem(dbc.NavLink("Create New", href="#", id="nav-create")),
            ], className="ms-auto", navbar=True),
        ]),
        color="dark",
        dark=True,
        className="mb-4",
    ),
    
    # Main container
    dbc.Container([
        # Main content area - initialize with projects layout
        html.Div(id="page-content", children=create_projects_layout(), className="p-4"),
    ], fluid=True),
    
    # Footer
    dbc.Container([
        html.Hr(),
        html.Footer([
            html.P("LLM Full-Stack Developer System", className="text-center text-muted"),
        ]),
    ], fluid=True),
])