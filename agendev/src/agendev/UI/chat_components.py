"""
Chat container components for the AgenDev UI.

This module contains utility functions for creating chat container elements
such as task sections, system messages, and markdown rendering.
"""
from typing import List, Union, Any, Dict, Optional, cast, Literal
from dash import html, dcc
from pydantic import BaseModel, Field

# Define a DashComponent type for better type hinting
DashComponent = Any  # This is safer than trying to reference a non-existent base class

def render_markdown(markdown_text: str) -> dcc.Markdown:
    """
    Render markdown content with custom styling.
    
    Args:
        markdown_text: Markdown text to render
        
    Returns:
        Dash Markdown component with rendered content
    """
    # Process checkboxes with custom rendering
    lines = markdown_text.split('\n')
    for i, line in enumerate(lines):
        if '- [ ]' in line:
            lines[i] = line.replace('- [ ]', '- <input type="checkbox"><span>')
            lines[i] += '</span>'
        elif '- [x]' in line:
            lines[i] = line.replace('- [x]', '- <input type="checkbox" checked><span>')
            lines[i] += '</span>'
    
    processed_markdown = '\n'.join(lines)
    
    # Return Markdown component with proper HTML handling
    return dcc.Markdown(
        processed_markdown,
        className="todo-markdown",
        dangerously_allow_html=True
    )

def create_system_message(title: str, content: Union[str, List[DashComponent]]) -> html.Div:
    """
    Create a system message component.
    
    Args:
        title: Message title
        content: Message content (string or list of components)
        
    Returns:
        Dash component for system message
    """
    return html.Div(
        className="system-message",
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.I(className="fas fa-robot", style={"fontSize": "24px", "marginRight": "10px", "color": "#61dafb"}),
                    html.Span(title, style={"fontSize": "24px", "fontWeight": "bold"})
                ]
            ),
            html.Div(content) if isinstance(content, str) else html.Div(children=content)
        ]
    )

def create_todo_display(todo_content: str) -> html.Div:
    """
    Create the todo.md display component.
    
    Args:
        todo_content: Markdown content for the todo list
        
    Returns:
        Dash component for todo display
    """
    return html.Div(
        id="todo-display",
        className="system-message",
        style={"marginTop": "20px", "marginBottom": "20px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "marginBottom": "15px"},
                children=[
                    html.I(className="fas fa-tasks", style={"fontSize": "18px", "marginRight": "10px", "color": "#61dafb"}),
                    html.Span("Project Tasks", style={"fontSize": "18px", "fontWeight": "bold"})
                ]
            ),
            html.Div(id="todo-content", children=render_markdown(todo_content))
        ]
    )

def create_collapsible_section(id_prefix: str, header_content: Union[html.Div, str], 
                              content: List[DashComponent], is_open: bool = True,
                              header_background: str = "#333") -> html.Div:
    """
    Create a collapsible section component.
    
    Args:
        id_prefix: Prefix for the section's HTML IDs
        header_content: Content for the section header
        content: Content for the collapsible body
        is_open: Whether the section is initially open
        header_background: Background color for the header (for different status indicators)
        
    Returns:
        Dash component for collapsible section
    """
    return html.Div([
        html.Div(
            className="collapsible-header",
            id=f"{id_prefix}-header",
            style={"backgroundColor": header_background},
            children=[
                html.I(
                    className="fas fa-chevron-down mr-2",
                    style={"marginRight": "10px"}
                ),
                header_content if isinstance(header_content, html.Div) else html.Div(header_content)
            ]
        ),
        html.Div(
            id=f"{id_prefix}-content",
            className="collapsible-content",
            style={"display": "block" if is_open else "none"},
            children=content
        )
    ])

def create_command_element(command: str, status: str = "completed") -> html.Div:
    """
    Create a command execution element.
    
    Args:
        command: Command text
        status: Execution status ("completed", "in-progress", or "failed")
        
    Returns:
        Dash component for command element
    """
    if status == "completed":
        icon_class = "fas fa-check-circle text-success"
        icon_color = "#00ff00"
    elif status == "in-progress":
        icon_class = "fas fa-spinner fa-spin text-warning"
        icon_color = "#ffc107"
    elif status == "failed":
        icon_class = "fas fa-times-circle text-danger"
        icon_color = "#dc3545"
    else:
        icon_class = "fas fa-circle text-secondary"
        icon_color = "#888"
    
    # Display the command as is
    display_command = command
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(html.I(className=icon_class), className="status-icon", style={"color": icon_color}),
            html.Span("Executing command", style={"marginRight": "10px"}),
            html.Code(display_command, className="command-element")
        ]
    )

def create_file_operation(operation: str, filepath: str, status: str = "completed") -> html.Div:
    """
    Create a file operation status element.
    
    Args:
        operation: Operation type ("Creating", "Editing", etc.)
        filepath: Path of the file being operated on
        status: Operation status ("completed", "in-progress", or "failed")
        
    Returns:
        Dash component for file operation status
    """
    if status == "completed":
        icon_class = "fas fa-check-circle text-success"
        icon_color = "#00ff00"
    elif status == "in-progress":
        icon_class = "fas fa-spinner fa-spin text-warning"
        icon_color = "#ffc107"
    elif status == "failed":
        icon_class = "fas fa-times-circle text-danger"
        icon_color = "#dc3545"
    else:
        icon_class = "fas fa-circle text-secondary"
        icon_color = "#888"
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(html.I(className=icon_class), className="status-icon", style={"color": icon_color}),
            html.Span(f"{operation} file", style={"marginRight": "10px"}),
            html.Code(filepath, className="file-path")
        ]
    )

def create_error_message(message: str, error_type: str = "error", details: Optional[str] = None) -> html.Div:
    """
    Create an error message component with detailed formatting.
    
    Args:
        message: The main error message
        error_type: Type of error ("error", "warning", or "info")
        details: Optional detailed error information
        
    Returns:
        Dash component for error message
    """
    if error_type == "error":
        icon_class = "fas fa-exclamation-circle"
        bg_color = "rgba(220, 53, 69, 0.1)"  # Soft red background
        border_color = "#dc3545"
        icon_color = "#dc3545"
    elif error_type == "warning":
        icon_class = "fas fa-exclamation-triangle"
        bg_color = "rgba(255, 193, 7, 0.1)"  # Soft yellow background
        border_color = "#ffc107"
        icon_color = "#ffc107"
    else:  # info
        icon_class = "fas fa-info-circle"
        bg_color = "rgba(97, 218, 251, 0.1)"  # Soft blue background
        border_color = "#61dafb"
        icon_color = "#61dafb"
    
    # Create the error message component
    children = [
        html.Div(
            style={
                "display": "flex", 
                "alignItems": "center", 
                "marginBottom": "10px"
            },
            children=[
                html.I(
                    className=icon_class, 
                    style={
                        "fontSize": "20px", 
                        "marginRight": "10px", 
                        "color": icon_color
                    }
                ),
                html.H4(
                    message, 
                    style={
                        "margin": "0", 
                        "fontSize": "16px", 
                        "fontWeight": "bold"
                    }
                )
            ]
        )
    ]
    
    # Add details if provided
    if details:
        children.append(
            html.Div(
                className="error-details",
                style={
                    "marginLeft": "30px", 
                    "marginTop": "5px", 
                    "padding": "8px", 
                    "backgroundColor": "rgba(0, 0, 0, 0.2)", 
                    "borderLeft": f"3px solid {border_color}", 
                    "borderRadius": "3px",
                    "fontFamily": "'Consolas', 'Courier New', monospace",
                    "fontSize": "0.9em",
                    "overflowX": "auto"
                },
                children=details
            )
        )
    
    return html.Div(
        className=f"error-message {error_type}-message",
        style={
            "padding": "15px",
            "backgroundColor": bg_color,
            "border": f"1px solid {border_color}",
            "borderRadius": "5px",
            "marginBottom": "15px"
        },
        children=children
    )

def create_task_metadata(
    metadata: Dict[str, Any], 
    style: Optional[Dict[str, str]] = None
) -> html.Div:
    """
    Create a component to display detailed task metadata.
    
    Args:
        metadata: Dictionary of task metadata
        style: Optional custom styling
        
    Returns:
        Dash component for task metadata
    """
    default_style = {
        "marginTop": "10px",
        "padding": "10px",
        "backgroundColor": "rgba(255, 255, 255, 0.05)",
        "borderRadius": "5px",
        "fontSize": "0.9em"
    }
    
    if style:
        default_style.update(style)
    
    # Create metadata items
    metadata_items = []
    for key, value in metadata.items():
        # Format the key for display
        display_key = key.replace('_', ' ').capitalize()
        
        # Format the value based on type
        if isinstance(value, (list, tuple)):
            display_value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            display_value = json.dumps(value, indent=2)
        else:
            display_value = str(value)
        
        metadata_items.append(
            html.Div(
                className="metadata-item",
                style={"marginBottom": "5px"},
                children=[
                    html.Span(
                        f"{display_key}: ", 
                        style={"fontWeight": "bold", "color": "#61dafb"}
                    ),
                    html.Span(display_value)
                ]
            )
        )
    
    return html.Div(
        className="task-metadata",
        style=default_style,
        children=metadata_items
    )

class TaskContent(BaseModel):
    """Model for task content with validation."""
    title: str
    description: Optional[str] = None
    status: str
    content_items: List[Dict[str, Any]]
    
    class Config:
        arbitrary_types_allowed = True

def create_task_status_badge(status: str) -> html.Span:
    """
    Create a status badge for tasks.
    
    Args:
        status: Task status ("completed", "in-progress", "failed", etc.)
        
    Returns:
        Dash component for status badge
    """
    # Define status colors and icons
    status_info = {
        "completed": {
            "bg_color": "#00ff00",
            "text_color": "#000",
            "icon": "fas fa-check-circle",
            "label": "Completed"
        },
        "in-progress": {
            "bg_color": "#ffc107",
            "text_color": "#000",
            "icon": "fas fa-spinner fa-spin",
            "label": "In Progress"
        },
        "failed": {
            "bg_color": "#dc3545",
            "text_color": "#fff",
            "icon": "fas fa-times-circle",
            "label": "Failed"
        },
        "planned": {
            "bg_color": "#888",
            "text_color": "#fff",
            "icon": "fas fa-circle",
            "label": "Planned"
        },
        "blocked": {
            "bg_color": "#6c757d",
            "text_color": "#fff",
            "icon": "fas fa-lock",
            "label": "Blocked"
        },
        "waiting_review": {
            "bg_color": "#17a2b8",
            "text_color": "#fff",
            "icon": "fas fa-clock",
            "label": "Waiting Review"
        }
    }
    
    # Use default if status not found
    info = status_info.get(status, {
        "bg_color": "#888",
        "text_color": "#fff",
        "icon": "fas fa-question-circle",
        "label": status.replace("_", " ").capitalize()
    })
    
    return html.Span(
        className="status-badge",
        style={
            "backgroundColor": info["bg_color"],
            "color": info["text_color"],
            "padding": "3px 8px",
            "borderRadius": "12px",
            "fontSize": "0.8em",
            "fontWeight": "bold",
            "display": "inline-flex",
            "alignItems": "center"
        },
        children=[
            html.I(
                className=info["icon"],
                style={"marginRight": "5px"}
            ),
            info["label"]
        ]
    )

def create_task_section(task_id: str, title: str, status: str, content: List[DashComponent]) -> html.Div:
    """
    Create a task section component with enhanced styling and status indicators.
    
    Args:
        task_id: Unique ID for the task
        title: Task title
        status: Task status ("completed", "in-progress", "failed", etc.)
        content: List of components for the task content
        
    Returns:
        Dash component for task section
    """
    # Determine header background color based on status
    header_colors = {
        "completed": "#2a3a2a",  # Dark green
        "in-progress": "#3a3a2a", # Dark yellow-ish
        "failed": "#3a2a2a",      # Dark red-ish
        "planned": "#2a2a3a",     # Dark blue-ish
        "blocked": "#333333"      # Dark gray
    }
    header_bg = header_colors.get(status, "#333")
    
    # Create header with icon, title and status badge
    header_content = html.Div(
        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "width": "100%"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center"},
                children=[
                    html.Span(title)
                ]
            ),
            create_task_status_badge(status)
        ]
    )
    
    # Create section with pattern-matching ID format
    return html.Div([
        html.Div(
            className="collapsible-header",
            id={"type": "task-header", "index": task_id},
            style={"backgroundColor": header_bg},
            children=[
                html.I(
                    className="fas fa-chevron-down mr-2",
                    style={"marginRight": "10px"}
                ),
                header_content
            ]
        ),
        html.Div(
            id={"type": "task-content", "index": task_id},
            className="collapsible-content",
            style={"display": "block" if status in ["in-progress", "failed"] else "none"},
            children=content
        )
    ])

def create_nested_collapsible(
    title: str, 
    content: List[DashComponent], 
    icon: str = "fas fa-info-circle",
    is_open: bool = False
) -> html.Div:
    """
    Create a nested collapsible section for detailed information.
    
    Args:
        title: Title for the collapsible section
        content: List of components for the section content
        icon: Icon class for the section
        is_open: Whether the section is initially open
        
    Returns:
        Dash component for nested collapsible section
    """
    # Generate a unique ID for this collapsible
    section_id = f"nested-{title.lower().replace(' ', '-')}-{id(title)}"
    
    return html.Div(
        className="nested-collapsible",
        style={"marginTop": "10px", "marginBottom": "10px"},
        children=[
            html.Div(
                className="nested-collapsible-header",
                id={"type": "nested-header", "index": section_id},
                style={
                    "backgroundColor": "rgba(255, 255, 255, 0.05)",
                    "padding": "8px 12px",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                    "display": "flex",
                    "alignItems": "center"
                },
                children=[
                    html.I(
                        className=f"{icon} mr-2",
                        style={"marginRight": "8px", "color": "#61dafb"}
                    ),
                    html.Span(
                        title,
                        style={"fontWeight": "bold", "fontSize": "0.9em"}
                    ),
                    html.I(
                        className="fas fa-chevron-down",
                        style={"marginLeft": "auto"}
                    )
                ]
            ),
            html.Div(
                id={"type": "nested-content", "index": section_id},
                className="nested-collapsible-content",
                style={
                    "display": "block" if is_open else "none",
                    "padding": "10px 15px",
                    "backgroundColor": "rgba(255, 255, 255, 0.02)",
                    "borderRadius": "0 0 4px 4px",
                    "marginTop": "1px"
                },
                children=content
            )
        ]
    )

def create_progress_indicator(progress: float, label: Optional[str] = None) -> html.Div:
    """
    Create a visual progress indicator.
    
    Args:
        progress: Progress value from 0 to 100
        label: Optional label to display
        
    Returns:
        Dash component for progress indicator
    """
    # Determine color based on progress
    if progress >= 80:
        color = "#00ff00"  # Green
    elif progress >= 40:
        color = "#ffc107"  # Yellow
    else:
        color = "#888"     # Gray
    
    progress_element = html.Div(
        className="progress-bar-container",
        style={
            "width": "100%",
            "backgroundColor": "rgba(255, 255, 255, 0.1)",
            "borderRadius": "4px",
            "height": "8px",
            "overflow": "hidden"
        },
        children=[
            html.Div(
                className="progress-bar",
                style={
                    "width": f"{progress}%",
                    "backgroundColor": color,
                    "height": "100%",
                    "transition": "width 0.5s ease-in-out"
                }
            )
        ]
    )
    
    if label:
        return html.Div(
            className="progress-indicator",
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "marginBottom": "5px"
                    },
                    children=[
                        html.Span(label, style={"fontSize": "0.9em"}),
                        html.Span(f"{progress:.1f}%", style={"fontSize": "0.9em"})
                    ]
                ),
                progress_element
            ]
        )
    else:
        return progress_element