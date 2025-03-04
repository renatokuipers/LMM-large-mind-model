import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import json
from typing import Dict, List, Optional
import uuid

# Initialize Dash app with dark theme and extra Bootstrap icons
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,  # Add Bootstrap icons
        "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap",  # Add custom font
    ],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},  # Ensure mobile responsiveness
    ],
)
server = app.server

# Common styles for consistency
CARD_STYLE = {
    "height": "100%",
    "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
    "transition": "transform 0.3s ease",
    "border": "none",
    "border-radius": "8px",
}

HOVER_STYLE = {
    "transform": "translateY(-5px)",
    "box-shadow": "0 8px 16px rgba(0, 0, 0, 0.3)",
}

BUTTON_STYLE = {
    "border-radius": "20px",
    "font-weight": "bold",
    "transition": "all 0.3s ease",
}

# Helper functions
def create_breadcrumbs(items):
    """Create a breadcrumb navigation component compatible with older dbc versions"""
    breadcrumb_items = []
    for i, (label, href, active) in enumerate(items):
        # Use dictionary format instead of BreadcrumbItem component
        if active:
            item = {"label": label, "active": True}
        else:
            item = {"label": label, "href": href}
        breadcrumb_items.append(item)
    
    return dbc.Breadcrumb(
        items=breadcrumb_items,
        class_name="mb-4 bg-light text-dark p-2 rounded"
    )

def create_empty_state(message, button_text=None, button_id=None):
    """Create a visually appealing empty state component"""
    children = [
        html.Div(
            html.I(className="bi bi-inbox fs-1 text-muted"),
            className="text-center mb-3"
        ),
        html.Div(message, className="text-center text-muted mb-3")
    ]
    
    if button_text and button_id:
        children.append(
            html.Div(
                dbc.Button(
                    [html.I(className="bi bi-plus-circle me-2"), button_text],
                    id=button_id,
                    color="primary",
                    style=BUTTON_STYLE
                ),
                className="text-center"
            )
        )
    
    return dbc.Card(
        dbc.CardBody(children),
        className="p-5",
        style=CARD_STYLE
    )

# Projects list page
def create_projects_layout(projects_data=None):
    """Create the projects list page with improved UI"""
    # Add breadcrumbs for navigation
    breadcrumbs = create_breadcrumbs([
        ("Home", "#", False),
        ("Projects", "#", True),
    ])
    
    # Handle empty state with a better visual
    if not projects_data or not projects_data.get('projects'):
        return dbc.Container([
            breadcrumbs,
            html.H2("Your Projects", className="mb-4"),
            create_empty_state(
                "No projects found. Start by creating your first project!",
                "Create New Project",
                "btn-create-project"
            )
        ])
    
    # Build project cards with improved design
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
        created_date = project.get('created_at', '').split('T')[0] if 'T' in project.get('created_at', '') else project.get('created_at', '')
        
        # Tech badges summary
        tech_stack = project.get('tech_stack', {})
        tech_badges = []
        
        if tech_stack.get('frontend'):
            tech_badges.append(dbc.Badge("Frontend", color="info", className="me-1 mb-1"))
        
        if tech_stack.get('backend'):
            tech_badges.append(dbc.Badge("Backend", color="success", className="me-1 mb-1"))
        
        if tech_stack.get('database'):
            tech_badges.append(dbc.Badge("Database", color="warning", className="me-1 mb-1"))
        
        # Create card for each project with hover effect
        card = dbc.Card([
            dbc.CardHeader([
                html.Div(
                    [html.I(className="bi bi-kanban me-2"), project.get('name', 'Unnamed Project')],
                    className="fw-bold fs-5"
                )
            ], className="d-flex justify-content-between align-items-center"),
            dbc.CardBody([
                html.P(
                    project.get('description', 'No description'),
                    className="card-text mb-3",
                    style={"height": "60px", "overflow": "hidden", "text-overflow": "ellipsis"}
                ),
                html.Div(tech_badges, className="mb-3"),
                html.Div([
                    html.Span(f"Progress: {completion:.1f}%", className="me-2"),
                    dbc.Progress(
                        value=completion,
                        className="mb-3",
                        style={"height": "10px"},
                        animated=True,
                        striped=True,
                    ),
                ]),
                html.Div([
                    html.Span(
                        [html.I(className="bi bi-calendar-date me-1"), f"Created: {created_date}"],
                        className="text-muted small"
                    ),
                    html.Span(
                        [html.I(className="bi bi-card-checklist me-1"), f"{completed_tasks}/{total_tasks} tasks"],
                        className="text-muted small ms-3"
                    ),
                ], className="d-flex mb-3"),
                dbc.Button(
                    [html.I(className="bi bi-eye me-2"), "View Project"],
                    id={"type": "btn-view-project", "index": project_id},
                    color="primary",
                    className="w-100",
                    style=BUTTON_STYLE
                ),
            ]),
        ],
        className="mb-4 h-100",
        style=CARD_STYLE,
        id={"type": "project-card", "index": project_id}
        )
        
        project_cards.append(dbc.Col(card, lg=4, md=6, sm=12, className="mb-4"))
    
    # Add "Create New Project" card
    new_project_card = dbc.Card([
        dbc.CardBody([
            html.Div(
                html.I(className="bi bi-plus-circle", style={"fontSize": "3rem"}),
                className="text-center text-muted mb-3"
            ),
            html.H5("Create New Project", className="text-center mb-3"),
            dbc.Button(
                "Start Building",
                id="btn-create-project",
                color="success",
                className="w-100",
                style=BUTTON_STYLE
            ),
        ]),
    ],
    className="mb-4 h-100 bg-light bg-opacity-10",
    style=CARD_STYLE
    )
    
    project_cards.append(dbc.Col(new_project_card, lg=4, md=6, sm=12, className="mb-4"))
    
    # Search and filter controls
    search_filter = dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText(html.I(className="bi bi-search")),
                dbc.Input(type="text", id="project-search", placeholder="Search projects..."),
            ]),
        ], lg=6, md=8, sm=12),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-sort-alpha-down me-2"), "Name"],
                    id="sort-by-name",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    [html.I(className="bi bi-sort-numeric-down me-2"), "Progress"],
                    id="sort-by-progress",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
                dbc.Button(
                    [html.I(className="bi bi-calendar me-2"), "Date"],
                    id="sort-by-date",
                    color="secondary",
                    outline=True,
                    size="sm",
                ),
            ]),
        ], lg=6, md=4, sm=12, className="d-flex justify-content-end align-items-center"),
    ], className="mb-4")
    
    return dbc.Container([
        breadcrumbs,
        html.H2(
            [html.I(className="bi bi-kanban-fill me-2"), "Your Projects"],
            className="mb-4 d-flex align-items-center"
        ),
        search_filter,
        dbc.Row(project_cards),
    ])

# Create new project form with validation
def create_new_project_layout():
    """Create project form with validation and better UI"""
    breadcrumbs = create_breadcrumbs([
        ("Home", "#", False),
        ("Projects", "#", False),
        ("Create New Project", "#", True),
    ])
    
    return dbc.Container([
        breadcrumbs,
        dbc.Card([
            dbc.CardHeader([
                html.H4(
                    [html.I(className="bi bi-plus-circle me-2"), "Create New Project"],
                    className="mb-0"
                )
            ]),
            dbc.CardBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Project Name", html_for="project-name", className="fw-bold"),
                            dbc.InputGroup([
                                dbc.InputGroupText(html.I(className="bi bi-bookmark")),
                                dbc.Input(
                                    type="text",
                                    id="project-name",
                                    placeholder="Enter project name",
                                    required=True,
                                    minLength=3,
                                    maxLength=100,
                                ),
                            ]),
                            dbc.FormFeedback("Please enter a project name (3-100 characters)", type="invalid"),
                            dbc.FormText("A clear, concise name for your project"),
                        ], md=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Project Description", html_for="project-description", className="fw-bold"),
                            dbc.InputGroup([
                                dbc.InputGroupText(html.I(className="bi bi-file-text")),
                                dbc.Textarea(
                                    id="project-description",
                                    placeholder="Describe your project in detail...",
                                    required=True,
                                    minLength=10,
                                    maxLength=1000,
                                    style={"height": "150px"}
                                ),
                            ]),
                            dbc.FormFeedback("Please enter a description (10-1000 characters)", type="invalid"),
                            dbc.FormText("A detailed description of what the project will do and its requirements"),
                        ]),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Output Directory", html_for="output-directory", className="fw-bold"),
                            dbc.InputGroup([
                                dbc.InputGroupText(html.I(className="bi bi-folder")),
                                dbc.Input(
                                    type="text",
                                    id="output-directory",
                                    placeholder="C:\\path\\to\\output\\directory",
                                    required=True,
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-folder2-open"),
                                    id="btn-browse-directory",
                                    color="secondary",
                                    n_clicks=0,
                                ),
                            ]),
                            dbc.FormFeedback("Please enter a valid directory path", type="invalid"),
                            dbc.FormText("Directory where project files will be generated"),
                        ], md=6),
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="form-error-message", className="text-danger mb-3"),
                            dbc.Button(
                                [html.I(className="bi bi-check-circle me-2"), "Create Project"],
                                id="btn-submit-project",
                                color="primary",
                                className="me-2",
                                style=BUTTON_STYLE,
                                n_clicks=0,
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-x-circle me-2"), "Cancel"],
                                id="btn-cancel-project",
                                color="secondary",
                                style=BUTTON_STYLE,
                                n_clicks=0,
                            ),
                        ], className="d-flex justify-content-end"),
                    ]),
                ], id="project-form"),
            ]),
        ], style=CARD_STYLE),
    ])

# Project detail view with enhanced UI
def create_project_detail_layout(project_id, project_data):
    """Create project detail view with enhanced UI/UX"""
    if not project_data:
        return dbc.Container([
            create_breadcrumbs([
                ("Home", "#", False),
                ("Projects", "#", False),
                ("Project Not Found", "#", True),
            ]),
            create_empty_state(
                "Project not found or has been deleted.",
                "Back to Projects",
                "btn-back-to-projects"
            )
        ])
    
    # Set up breadcrumbs for navigation
    breadcrumbs = create_breadcrumbs([
        ("Home", "#", False),
        ("Projects", "#", False),
        (project_data.get('name', 'Unnamed Project'), "#", True),
    ])
    
    # Extract epics and tasks
    epics = project_data.get('epics', [])
    
    # Create epic accordion items with improved UI
    epic_items = []
    for i, epic in enumerate(epics):
        # Calculate epic progress
        total_tasks = len(epic.get('tasks', []))
        completed_tasks = sum(1 for task in epic.get('tasks', []) if task.get('status') == 'completed')
        progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Status indicator
        status_indicators = {
            "planned": {"color": "secondary", "icon": "bi-hourglass"},
            "in_progress": {"color": "primary", "icon": "bi-arrow-repeat"},
            "completed": {"color": "success", "icon": "bi-check-circle"},
            "failed": {"color": "danger", "icon": "bi-exclamation-circle"}
        }
        
        # Determine epic status
        epic_status = "planned"
        if all(task.get('status') == 'completed' for task in epic.get('tasks', [])):
            epic_status = "completed"
        elif any(task.get('status') == 'in_progress' for task in epic.get('tasks', [])):
            epic_status = "in_progress"
        elif any(task.get('status') == 'failed' for task in epic.get('tasks', [])):
            epic_status = "failed"
        
        status_indicator = status_indicators.get(epic_status, status_indicators["planned"])
        
        # Create tasks table with enhanced UI
        tasks_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th(html.I(className="bi bi-list-task"), style={"width": "5%"}),
                    html.Th("Task", style={"width": "45%"}),
                    html.Th("Status", style={"width": "20%"}),
                    html.Th("Code Items", style={"width": "15%"}),
                    html.Th("Actions", style={"width": "15%"}),
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(i+1),
                    html.Td([
                        html.Div(task.get('title', 'Unnamed Task'), className="fw-bold"),
                        html.Small(
                            task.get('description', '')[:50] + ('...' if len(task.get('description', '')) > 50 else ''),
                            className="text-muted d-block"
                        ),
                    ]),
                    html.Td([
                        dbc.Badge(
                            [
                                html.I(className=f"bi {status_indicators.get(task.get('status', 'planned'), status_indicators['planned'])['icon']} me-1"),
                                task.get('status', 'planned').replace('_', ' ').upper()
                            ],
                            color=status_indicators.get(task.get('status', 'planned'), status_indicators['planned'])['color'],
                            className="text-uppercase",
                            pill=True,
                        )
                    ]),
                    html.Td([
                        dbc.Badge(
                            str(len(task.get('code_items', []))),
                            color="info",
                            pill=True,
                            className="px-2"
                        ) if task.get('code_items') else "None"
                    ]),
                    html.Td([
                        dbc.ButtonGroup([
                            dbc.Button(
                                html.I(className="bi bi-info-circle"),
                                id={"type": "btn-task-details", "epic": epic.get('id'), "task": task.get('id')},
                                color="info",
                                size="sm",
                                className="me-1",
                                title="View Details",
                                outline=True,
                            ),
                            dbc.Button(
                                html.I(className="bi bi-code-slash"),
                                id={"type": "btn-generate-task", "epic": epic.get('id'), "task": task.get('id')},
                                color="primary",
                                size="sm",
                                disabled=task.get('status') == 'completed',
                                title="Generate Code",
                                outline=True,
                            ),
                        ])
                    ]),
                ], className=f"table-{'success' if task.get('status') == 'completed' else 'default'}-subtle")
                for j, task in enumerate(epic.get('tasks', []))
            ]),
        ], bordered=True, hover=True, responsive=True, size="sm", dark=True, striped=True)
        
        # Create epic card with better visual hierarchy
        epic_item = dbc.AccordionItem(
            [
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Description", className="text-muted mb-2"),
                                    html.P(epic.get('description', 'No description')),
                                ])
                            ], className="mb-3 bg-dark bg-opacity-25"),
                            
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Progress", className="text-muted mb-2"),
                                    html.Div([
                                        html.Div(
                                            [
                                                html.Span(f"{progress:.1f}%", className="fs-4 fw-bold"),
                                                html.Span(f" ({completed_tasks}/{total_tasks} tasks)", className="text-muted ms-2")
                                            ],
                                            className="mb-2"
                                        ),
                                        dbc.Progress(
                                            value=progress,
                                            className="mb-0",
                                            style={"height": "10px"},
                                            animated=True,
                                            striped=True,
                                            color=status_indicator["color"]
                                        ),
                                    ]),
                                ])
                            ], className="mb-3 bg-dark bg-opacity-25"),
                        ], md=4),
                        
                        dbc.Col([
                            tasks_table,
                        ], md=8),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-code me-2"), "Generate Code for Epic"],
                                id={"type": "btn-generate-epic", "index": epic.get('id')},
                                color="primary",
                                className="me-2",
                                style=BUTTON_STYLE,
                                disabled=epic_status == "completed"
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-pencil me-2"), "Edit Epic"],
                                id={"type": "btn-edit-epic", "index": epic.get('id')},
                                color="secondary",
                                outline=True,
                                className="me-2",
                                style=BUTTON_STYLE,
                            ),
                        ]),
                    ]),
                ]),
            ],
            title=[
                html.Div(
                    [
                        html.I(className=f"bi {status_indicator['icon']} me-2", style={"color": f"var(--bs-{status_indicator['color']})"}, id={"type": "epic-status-icon", "index": i}),
                        f"Epic {i+1}: {epic.get('title', 'Unnamed Epic')}",
                    ],
                    className="d-flex align-items-center"
                )
            ],
            item_id=epic.get('id', f"epic-{i}")
        )
        
        epic_items.append(epic_item)
    
    # Tech stack badges with improved visuals
    tech_stack = project_data.get('tech_stack', {})
    tech_sections = []
    
    tech_icons = {
        "frontend": "bi-window",
        "backend": "bi-server",
        "database": "bi-database"
    }
    
    for category, icon in tech_icons.items():
        if tech_stack.get(category):
            tech_badges = [
                dbc.Badge(tech, color={"frontend": "info", "backend": "success", "database": "warning"}[category], className="me-1 mb-1")
                for tech in tech_stack.get(category, [])
            ]
            
            tech_sections.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H6([html.I(className=f"bi {icon} me-2"), category.capitalize()], className="mb-2"),
                        html.Div(tech_badges) if tech_badges else html.Div("None specified", className="text-muted")
                    ])
                ], className="mb-3 h-100 bg-dark bg-opacity-25")
            )
    
    # Calculate overall project progress
    total_tasks = 0
    completed_tasks = 0
    for epic in epics:
        for task in epic.get('tasks', []):
            total_tasks += 1
            if task.get('status') == 'completed':
                completed_tasks += 1
    
    overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Project header with summary metrics
    project_header = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2(project_data.get('name', 'Unnamed Project'), className="mb-2"),
                    html.P(project_data.get('description', 'No description'), className="lead mb-3"),
                ], md=8),
                
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{overall_progress:.1f}%", className="text-center mb-0"),
                                    html.P("Completion", className="text-center text-muted mb-0"),
                                ], className="py-2")
                            ], className="text-center h-100 bg-dark bg-opacity-25")
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(str(len(epics)), className="text-center mb-0"),
                                    html.P("Epics", className="text-center text-muted mb-0"),
                                ], className="py-2")
                            ], className="text-center h-100 bg-dark bg-opacity-25")
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{completed_tasks}/{total_tasks}", className="text-center mb-0"),
                                    html.P("Tasks", className="text-center text-muted mb-0"),
                                ], className="py-2")
                            ], className="text-center h-100 bg-dark bg-opacity-25")
                        ], width=4),
                    ]),
                ], md=4),
            ]),
        ])
    ], className="mb-4 bg-dark bg-opacity-10", style=CARD_STYLE)
    
    # Action buttons
    action_buttons = dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="bi bi-arrow-left me-2"), "Back to Projects"],
                    id="btn-back-to-projects",
                    color="secondary",
                    outline=True,
                    className="me-2",
                    style=BUTTON_STYLE,
                ),
                dbc.Button(
                    [html.I(className="bi bi-code-square me-2"), "Generate All Code"],
                    id="btn-generate-all",
                    color="success",
                    className="me-2",
                    style=BUTTON_STYLE,
                    disabled=overall_progress == 100,
                ),
                dbc.Button(
                    [html.I(className="bi bi-folder2-open me-2"), "Open Output Directory"],
                    id="btn-open-directory",
                    color="info",
                    style=BUTTON_STYLE,
                ),
            ]),
        ], className="mb-4 d-flex"),
    ])
    
    # Tech stack section
    tech_stack_section = dbc.Row([
        dbc.Col(
            tech_section,
            md=4,
            sm=12,
        ) for tech_section in tech_sections
    ], className="mb-4") if tech_sections else None
    
    # Put it all together
    return dbc.Container([
        breadcrumbs,
        project_header,
        action_buttons,
        tech_stack_section if tech_stack_section else None,
        dbc.Row([
            dbc.Col([
                html.H4([html.I(className="bi bi-kanban-fill me-2"), "Project Plan"], className="mb-3"),
                dbc.Accordion(
                    epic_items,
                    start_collapsed=False,
                    always_open=True,
                    flush=True,
                    style={"border-radius": "8px", "overflow": "hidden"}
                ) if epic_items else create_empty_state("No epics defined for this project."),
            ]),
        ]),
    ])

# Task detail modal with enhanced UI
def create_task_detail_modal(epic_id, task_id, project_data):
    """Create a more functional and visually appealing task detail modal"""
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
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle text-danger fs-1"),
                ], className="text-center mb-3"),
                html.P("Task not found or has been deleted.", className="text-center"),
            ]),
            dbc.ModalFooter(dbc.Button("Close", id="close-task-modal", className="ms-auto")),
        ], id="task-detail-modal", is_open=True, size="lg", backdrop="static")
    
    # Status badge with icon
    status_indicators = {
        "planned": {"color": "secondary", "icon": "bi-hourglass", "text": "PLANNED"},
        "in_progress": {"color": "primary", "icon": "bi-arrow-repeat", "text": "IN PROGRESS"},
        "completed": {"color": "success", "icon": "bi-check-circle", "text": "COMPLETED"},
        "failed": {"color": "danger", "icon": "bi-exclamation-circle", "text": "FAILED"}
    }
    
    task_status = task.get('status', 'planned')
    status_indicator = status_indicators.get(task_status, status_indicators["planned"])
    status_badge = dbc.Badge(
        [
            html.I(className=f"bi {status_indicator['icon']} me-1"),
            status_indicator["text"]
        ],
        color=status_indicator["color"],
        className="fs-6 mb-3"
    )
    
    # Calculate progress
    code_items = task.get('code_items', [])
    total_items = len(code_items)
    implemented_items = sum(1 for item in code_items if item.get('implemented', False))
    progress = (implemented_items / total_items * 100) if total_items > 0 else 0
    
    # Create code items table with enhanced UI
    code_items_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th(html.I(className="bi bi-code-slash"), style={"width": "5%"}),
                html.Th("Name", style={"width": "20%"}),
                html.Th("Type", style={"width": "15%"}),
                html.Th("File", style={"width": "30%"}),
                html.Th("Status", style={"width": "15%"}),
                html.Th("Actions", style={"width": "15%"}),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(i+1),
                html.Td(item.get('name', 'Unnamed')),
                html.Td(
                    dbc.Badge(
                        item.get('type', 'Unknown'),
                        color={"Function": "info", "Class": "primary", "Method": "warning"}.get(item.get('type', ''), "secondary"),
                        pill=True,
                        className="px-2"
                    )
                ),
                html.Td([
                    html.Div(
                        [html.I(className="bi bi-file-earmark-code me-1"), item.get('file_path', 'Not specified')],
                        className="text-truncate",
                        style={"max-width": "100%"}
                    ) if item.get('file_path') else "Not specified"
                ]),
                html.Td(
                    dbc.Badge(
                        [
                            html.I(className=f"bi {'bi-check-circle' if item.get('implemented', False) else 'bi-hourglass'} me-1"),
                            "Implemented" if item.get('implemented', False) else "Pending"
                        ],
                        color="success" if item.get('implemented', False) else "secondary",
                        pill=True,
                    )
                ),
                html.Td(
                    dbc.Button(
                        html.I(className="bi bi-eye"),
                        id={"type": "btn-view-code", "epic": epic_id, "task": task_id, "item": i},
                        color="info",
                        size="sm",
                        outline=True,
                        disabled=not item.get('implemented', False),
                        title="View Code"
                    )
                ),
            ]) for i, item in enumerate(code_items)
        ]),
    ], bordered=True, hover=True, responsive=True, size="sm", dark=True, striped=True) if code_items else None
    
    # Format dates
    created_date = task.get('created_at', '').split('.')[0].replace('T', ' ') if 'T' in task.get('created_at', '') else task.get('created_at', 'Unknown')
    updated_date = task.get('updated_at', '').split('.')[0].replace('T', ' ') if 'T' in task.get('updated_at', '') else task.get('updated_at', 'Unknown')
    
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle([
                html.I(className=f"bi {status_indicator['icon']} me-2", style={"color": f"var(--bs-{status_indicator['color']})"}, id={"type": "task-status-icon", "index": task_id}),
                f"Task: {task.get('title', 'Unnamed Task')}"
            ])
        ], close_button=True),
        dbc.ModalBody([
            # Epic info
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Epic: ", className="text-muted"),
                        html.Span(epic.get('title', 'Unknown Epic')),
                    ], className="mb-3"),
                ], md=8),
                dbc.Col([
                    status_badge
                ], md=4, className="text-end"),
            ], className="mb-3"),
            
            # Progress bar
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(
                            [
                                html.Span("Progress: ", className="text-muted"),
                                html.Span(f"{progress:.1f}% ({implemented_items}/{total_items} items)"),
                            ],
                            className="mb-2"
                        ),
                        dbc.Progress(
                            value=progress,
                            className="mb-3",
                            style={"height": "8px"},
                            animated=True,
                            striped=True,
                            color=status_indicator["color"]
                        ),
                    ]),
                ]),
            ]),
            
            # Task content tabs
            dbc.Tabs([
                dbc.Tab([
                    html.Div(
                        task.get('description', 'No description'),
                        className="p-3 bg-dark bg-opacity-10 rounded"
                    )
                ], tab_id="tab-description", label=[html.I(className="bi bi-file-text me-2"), "Description"], className="mt-3"),
                
                dbc.Tab([
                    html.Div(
                        code_items_table if code_items else html.P("No code items defined for this task.", className="text-muted fst-italic text-center p-3"),
                        className="py-3"
                    )
                ], tab_id="tab-code-items", label=[html.I(className="bi bi-code-square me-2"), "Code Items"], className="mt-3"),
                
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6([html.I(className="bi bi-calendar-plus me-2"), "Created"]),
                                        html.P(created_date),
                                    ])
                                ], className="mb-3 bg-dark bg-opacity-10"),
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6([html.I(className="bi bi-calendar-check me-2"), "Last Updated"]),
                                        html.P(updated_date),
                                    ])
                                ], className="mb-3 bg-dark bg-opacity-10"),
                            ], md=6),
                        ]),
                        
                        dbc.Card([
                            dbc.CardBody([
                                html.H6([html.I(className="bi bi-link-45deg me-2"), "Dependencies"]),
                                html.P("None" if not task.get('dependencies') else ", ".join(task.get('dependencies'))),
                            ])
                        ], className="mb-3 bg-dark bg-opacity-10"),
                    ], className="p-3")
                ], tab_id="tab-info", label=[html.I(className="bi bi-info-circle me-2"), "Info"], className="mt-3"),
            ], id="task-tabs", active_key="tab-description"),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                [html.I(className="bi bi-code-slash me-2"), "Generate Code"],
                id={"type": "btn-generate-task", "epic": epic_id, "task": task_id},
                color="primary",
                className="me-2",
                style=BUTTON_STYLE,
                disabled=task_status == "completed"
            ),
            dbc.Button(
                [html.I(className="bi bi-pencil me-2"), "Edit Task"],
                id={"type": "btn-edit-task", "epic": epic_id, "task": task_id},
                color="secondary",
                outline=True, 
                className="me-2",
                style=BUTTON_STYLE,
            ),
            dbc.Button(
                [html.I(className="bi bi-x-lg me-2"), "Close"],
                id="close-task-modal",
                color="light",
                style=BUTTON_STYLE,
            ),
        ]),
    ], id="task-detail-modal", is_open=True, size="lg", backdrop="static", scrollable=True)

# Code preview modal
def create_code_preview_modal(epic_id, task_id, item_index, project_data):
    """Create a modal for previewing generated code"""
    # Find code item
    epic = None
    task = None
    code_item = None
    
    for e in project_data.get('epics', []):
        if e.get('id') == epic_id:
            epic = e
            for t in e.get('tasks', []):
                if t.get('id') == task_id:
                    task = t
                    if 0 <= item_index < len(t.get('code_items', [])):
                        code_item = t.get('code_items', [])[item_index]
                    break
            break
    
    if not code_item:
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody("Code item not found"),
            dbc.ModalFooter(dbc.Button("Close", id="close-code-preview", className="ms-auto")),
        ], id="code-preview-modal", is_open=True)
    
    # Try to read the file
    file_path = code_item.get('file_path')
    code_content = "// Code content not available"
    
    if file_path and os.path.exists(os.path.join(project_data.get('output_directory', ''), file_path)):
        try:
            with open(os.path.join(project_data.get('output_directory', ''), file_path), 'r') as f:
                code_content = f.read()
        except Exception as e:
            code_content = f"// Error reading file: {str(e)}"
    
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle([
                html.I(className="bi bi-code-slash me-2"),
                f"{code_item.get('type', 'Code')}: {code_item.get('name', 'Unnamed')}"
            ])
        ]),
        dbc.ModalBody([
            html.Div([
                html.Span("File: ", className="text-muted"),
                html.Code(file_path or "Not specified"),
            ], className="mb-3"),
            
            html.Div([
                html.Span("Description: ", className="text-muted"),
                html.Span(code_item.get('description', 'No description')),
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.Pre(
                        html.Code(
                            code_content,
                            className="language-python"
                        ),
                        className="mb-0",
                        style={"max-height": "400px", "overflow-y": "auto"}
                    )
                ])
            ], className="bg-dark"),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                [html.I(className="bi bi-clipboard me-2"), "Copy to Clipboard"],
                id="copy-code-button",
                color="secondary",
                outline=True,
                className="me-2",
            ),
            dbc.Button(
                [html.I(className="bi bi-pencil me-2"), "Edit"],
                id={"type": "btn-edit-code", "epic": epic_id, "task": task_id, "item": item_index},
                color="primary",
                outline=True,
                className="me-2",
            ),
            dbc.Button(
                "Close",
                id="close-code-preview",
                color="secondary",
            ),
        ]),
    ], id="code-preview-modal", is_open=True, size="lg", scrollable=True)

# Loading screen with status updates
def create_loading_screen(title, message=None):
    """Create a loading screen with status updates"""
    return dbc.Container([
        html.Div([
            html.H2(title, className="mb-4 text-center"),
            html.Div([
                dbc.Spinner(size="lg", color="primary", type="grow", className="me-2"),
                dbc.Spinner(size="lg", color="primary", type="grow", className="me-2"),
                dbc.Spinner(size="lg", color="primary", type="grow"),
            ], className="text-center mb-4"),
            html.Div(
                message or "Please wait, this may take a few moments...",
                id="loading-status",
                className="text-center text-muted fs-5"
            ),
            dbc.Progress(
                id="loading-progress",
                value=0,
                striped=True,
                animated=True,
                className="mt-4",
                style={"height": "10px"}
            ),
        ], className="py-5")
    ])

# Error page
def create_error_page(title, message, error_details=None):
    """Create an error page with details"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-exclamation-triangle-fill text-danger", style={"fontSize": "4rem"}),
                        ], className="text-center mb-3"),
                        html.H2(title, className="text-center mb-3"),
                        html.P(message, className="text-center mb-4"),
                        html.Div([
                            dbc.Button(
                                [html.I(className="bi bi-arrow-left me-2"), "Back to Projects"],
                                id="btn-back-to-projects",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-arrow-clockwise me-2"), "Try Again"],
                                id="btn-try-again",
                                color="secondary",
                            ),
                        ], className="text-center"),
                        
                        html.Div([
                            dbc.Collapse([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Pre(error_details, className="mb-0 text-danger", style={"whiteSpace": "pre-wrap"})
                                    ])
                                ], className="mt-4 bg-dark")
                            ], id="error-details-collapse"),
                            
                            dbc.Button(
                                html.Span([
                                    html.I(className="bi bi-info-circle me-2"),
                                    "Show Technical Details",
                                ], id="show-details-text"),
                                id="toggle-error-details",
                                color="link",
                                className="mt-3",
                            ) if error_details else None,
                        ]),
                    ])
                ], style=CARD_STYLE, className="mt-5")
            ], lg=8, className="mx-auto")
        ])
    ])

# App layout
app.layout = html.Div([
    # Stores
    dcc.Store(id='projects-store', storage_type='local', data={"projects": {}}),
    dcc.Store(id='current-project-id', storage_type='local'),
    dcc.Store(id='current-epic-id', storage_type='local'),
    dcc.Store(id='form-validation', storage_type='memory'),
    dcc.Store(id='toast-message', storage_type='memory'),
    
    # Toast container for notifications
    dbc.Toast(
        id="notification-toast",
        header="Notification",
        is_open=False,
        dismissable=True,
        duration=4000,
        icon="primary",
        style={"position": "fixed", "top": 66, "right": 10, "width": 350, "z-index": 1999},
    ),
    
    # Navigation bar with improved styling
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="bi bi-braces-asterisk me-2"),
                "LLM Full-Stack Developer"
            ], className="ms-2 fw-bold"),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink([html.I(className="bi bi-kanban me-2"), "Projects"], href="#", id="nav-projects")),
                    dbc.NavItem(dbc.NavLink([html.I(className="bi bi-plus-circle me-2"), "Create New"], href="#", id="nav-create")),
                ], className="ms-auto", navbar=True),
            ], id="navbar-collapse", navbar=True),
        ]),
        color="dark",
        dark=True,
        className="mb-4 shadow-sm",
        sticky="top",
    ),
    
    # Main container
    dbc.Container([
        # Main content area - initialize with projects layout
        html.Div(id="page-content", children=create_projects_layout(), className="p-2 p-md-4"),
    ], fluid=True),
    
    # Modals
    html.Div(id="task-detail-modal"),  # Placeholder for task detail modal
    html.Div(id="code-preview-modal"),  # Placeholder for code preview modal
    
    # Footer
    dbc.Container([
        html.Hr(),
        html.Footer([
            dbc.Row([
                dbc.Col([
                    html.P("LLM Full-Stack Developer System", className="text-center text-muted mb-0"),
                ], md=6),
                dbc.Col([
                    html.P([
                        html.I(className="bi bi-github me-2"),
                        html.A("View Source", href="#", className="text-muted")
                    ], className="text-center mb-0"),
                ], md=6),
            ]),
        ], className="py-3"),
    ], fluid=True),
])