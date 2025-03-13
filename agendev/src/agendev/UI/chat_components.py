"""
Chat container components for the AgenDev UI.

This module contains utility functions for creating chat container elements
such as task sections, system messages, and markdown rendering.
"""
from typing import List, Union, Any, Dict, Optional, cast
from dash import html, dcc
from pydantic import BaseModel

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
                              content: List[DashComponent], is_open: bool = True) -> html.Div:
    """
    Create a collapsible section component.
    
    Args:
        id_prefix: Prefix for the section's HTML IDs
        header_content: Content for the section header
        content: Content for the collapsible body
        is_open: Whether the section is initially open
        
    Returns:
        Dash component for collapsible section
    """
    return html.Div([
        html.Div(
            className="collapsible-header",
            id=f"{id_prefix}-header",
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
        status: Execution status ("completed" or "in-progress")
        
    Returns:
        Dash component for command element
    """
    icon_class = ("fas fa-check-circle text-success" if status == "completed" 
                else "fas fa-spinner fa-spin text-warning")
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span("Executing command", style={"marginRight": "10px"}),
            html.Code(command, className="command-element")
        ]
    )

def create_file_operation(operation: str, filepath: str, status: str = "completed") -> html.Div:
    """
    Create a file operation status element.
    
    Args:
        operation: Operation type ("Creating", "Editing", etc.)
        filepath: Path of the file being operated on
        status: Operation status ("completed" or "in-progress")
        
    Returns:
        Dash component for file operation status
    """
    icon_class = ("fas fa-check-circle text-success" if status == "completed" 
                else "fas fa-spinner fa-spin text-warning")
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span(f"{operation} file", style={"marginRight": "10px"}),
            html.Code(filepath, className="file-path")
        ]
    )

class TaskContent(BaseModel):
    """Model for task content with validation."""
    title: str
    description: Optional[str] = None
    status: str
    content_items: List[Dict[str, Any]]
    
    class Config:
        arbitrary_types_allowed = True

def create_task_section(task_id: str, title: str, status: str, content: List[DashComponent]) -> html.Div:
    """
    Create a task section component.
    
    Args:
        task_id: Unique ID for the task
        title: Task title
        status: Task status ("completed", "in-progress", or "pending")
        content: List of components for the task content
        
    Returns:
        Dash component for task section
    """
    # Determine icon class based on status
    if status == "completed":
        icon_class = "fas fa-check-circle"
        icon_style = {"marginRight": "10px", "color": "#00ff00"}
    elif status == "in-progress":
        icon_class = "fas fa-spinner fa-spin"
        icon_style = {"marginRight": "10px", "color": "#ffc107"}
    else:
        icon_class = "fas fa-circle"
        icon_style = {"marginRight": "10px", "color": "#888"}
    
    # Create header with icon and title
    header_content = html.Div([
        html.I(className=icon_class, style=icon_style),
        html.Span(title)
    ])
    
    # Create section with pattern-matching ID format
    return html.Div([
        html.Div(
            className="collapsible-header",
            id={"type": "task-header", "index": task_id},
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
            style={"display": "block" if status == "in-progress" else "none"},
            children=content
        )
    ])