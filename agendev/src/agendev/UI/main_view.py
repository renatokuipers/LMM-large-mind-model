"""
Main view module for the AgenDev UI.

This module contains the layout for the main view with chat and view containers.
"""
from dash import html, dcc
from typing import Dict, Any, List, Optional

from .chat_components import create_system_message, create_todo_display
from .view_components import (
    create_view_header, 
    create_view_type_indicator, 
    create_playback_controls
)

def create_chat_container(project_title: str, todo_content: str = "", task_sections: Optional[List] = None) -> html.Div:
    """
    Create the chat container layout.
    
    Args:
        project_title: Title of the current project
        todo_content: Markdown content for the todo list
        task_sections: List of task section components
        
    Returns:
        Dash component for chat container
    """
    return html.Div(
        className="chat-container",
        id="chat-container",
        children=[
            # Header with current task
            create_system_message(
                "AgenDev",
                html.H3(id="project-title", children=project_title, style={"margin": "0 0 10px 0"})
            ),
            
            # Todo.md display area
            create_todo_display(todo_content),
            
            # Task sections
            html.Div(id="task-sections", children=task_sections or [])
        ]
    )

def create_view_container() -> html.Div:
    """
    Create the view container layout.
    
    Returns:
        Dash component for view container
    """
    return html.Div(
        className="view-container",
        children=[
            # Header
            create_view_header(),
            
            # View type indicator
            create_view_type_indicator(),
            
            # Content area (can be terminal, editor, or browser)
            html.Div(
                className="view-content",
                id="view-content",
                children=[]  # Will be dynamically populated
            ),
            
            # Playback Controls
            create_playback_controls()
        ]
    )

def create_main_view(project_title: str = "Project Development", 
                    todo_content: str = "", 
                    task_sections: Optional[List] = None) -> html.Div:
    """
    Create the main view layout.
    
    Args:
        project_title: Title of the current project
        todo_content: Markdown content for the todo list
        task_sections: List of task section components
        
    Returns:
        Dash component for main view
    """
    return html.Div(
        id="main-container",
        className="main-container",
        style={"display": "none"},  # Hidden initially
        children=[
            # Left side - Chat interface
            create_chat_container(project_title, todo_content, task_sections),
            
            # Right side - Dynamic view (Terminal, Editor, Browser)
            create_view_container()
        ]
    )

def create_stores() -> List[dcc.Store]:
    """
    Create store components for the application state.
    
    Returns:
        List of Dash Store components
    """
    return [
        # App state store
        dcc.Store(
            id='app-state',
            data={
                "view": "landing",
                "initial_prompt": "",
                "current_task_index": 0,
                "is_live_mode": True
            }
        ),
        
        # Store for playback data
        dcc.Store(
            id='playback-data',
            data={
                "total_steps": 0,
                "current_step": 0,
                "is_playing": False,
                "play_interval": 3,  # seconds between steps
                "steps": []
            }
        ),
        
        # Store for todo.md content
        dcc.Store(
            id='todo-data',
            data={
                "content": "# Project Development\n## Setup Phase\n- [ ] Initialize project directory\n- [ ] Set up development environment\n## Implementation Phase\n- [ ] Implement core functionality\n- [ ] Create user interface\n## Testing Phase\n- [ ] Write unit tests\n- [ ] Perform integration testing"
            }
        ),
        
        # Store for task sections
        dcc.Store(
            id='task-data',
            data={
                "tasks": [
                    {
                        "id": "task1",
                        "title": "Initialize project directory",
                        "status": "in-progress",
                        "content": []  # Populated at runtime
                    }
                ]
            }
        ),
        
        # Interval for playback
        dcc.Interval(
            id='playback-interval',
            interval=3000,  # 3 seconds between steps
            n_intervals=0,
            disabled=True
        )
    ]