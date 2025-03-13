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
    create_playback_controls,
    create_terminal_view
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

def create_view_container(
    view_content: Optional[List] = None,
    view_type: str = "Terminal",
    operation_type: str = "Setting up",
    file_path: str = "environment",
    is_replay: bool = False,
    current_step: int = 0,
    total_steps: int = 0,
    is_playing: bool = False,
    is_live: bool = True
) -> html.Div:
    """
    Create the enhanced view container layout with replay support.
    
    Args:
        view_content: Optional content for the view
        view_type: Current view type (Terminal, Editor, Browser)
        operation_type: Current operation being performed
        file_path: Path of the file being operated on
        is_replay: Whether in replay mode
        current_step: Current step index in replay mode
        total_steps: Total number of steps in replay mode
        is_playing: Whether playback is active
        is_live: Whether in live mode
        
    Returns:
        Dash component for view container
    """
    # Default content if none provided
    if view_content is None:
        view_content = [
            create_terminal_view(
                "$ echo 'Initializing AgenDev...'\nInitializing AgenDev...\n$ mkdir -p src tests docs\n"
            )
        ]
    
    return html.Div(
        className="view-container",
        children=[
            # Header
            create_view_header("AgenDev's Computer"),
            
            # View type indicator with replay information
            create_view_type_indicator(
                view_type=view_type,
                operation_type=operation_type,
                file_path=file_path,
                is_replay=is_replay,
                step_number=current_step + 1 if is_replay else None,
                total_steps=total_steps if is_replay else None
            ),
            
            # Content area (can be terminal, editor, or browser)
            html.Div(
                className="view-content",
                id="view-content",
                children=view_content
            ),
            
            # Playback Controls with enhanced replay support
            create_playback_controls(
                total_steps=total_steps,
                current_step=current_step,
                is_playing=is_playing,
                is_live=is_live
            )
        ]
    )

def create_main_view(
    project_title: str = "Project Development", 
    todo_content: str = "", 
    task_sections: Optional[List] = None,
    view_props: Optional[Dict[str, Any]] = None
) -> html.Div:
    """
    Create the main view layout with enhanced replay support.
    
    Args:
        project_title: Title of the current project
        todo_content: Markdown content for the todo list
        task_sections: List of task section components
        view_props: Properties for the view container
        
    Returns:
        Dash component for main view
    """
    # Default view properties if none provided
    if view_props is None:
        view_props = {
            "view_type": "Terminal",
            "operation_type": "Setting up",
            "file_path": "environment",
            "is_replay": False,
            "current_step": 0,
            "total_steps": 0,
            "is_playing": False,
            "is_live": True
        }
    
    return html.Div(
        id="main-container",
        className="main-container",
        style={"display": "none"},  # Hidden initially
        children=[
            # Left side - Chat interface
            create_chat_container(project_title, todo_content, task_sections),
            
            # Right side - Dynamic view (Terminal, Editor, Browser) with replay support
            create_view_container(**view_props)
        ]
    )

def create_replay_state_store() -> dcc.Store:
    """
    Create the replay state store for tracking playback state.
    
    Returns:
        Dash Store component for replay state
    """
    return dcc.Store(
        id='replay-state',
        data={
            "is_replay": False,
            "is_playing": False,
            "playback_speed": 1,
            "current_step": 0,
            "total_steps": 0,
            "step_timestamps": [],
            "step_types": [],
            "step_tooltips": [],
            "history": []  # For undo/redo functionality
        }
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
        
        # New store for replay state
        create_replay_state_store(),
        
        # Interval for playback
        dcc.Interval(
            id='playback-interval',
            interval=3000,  # 3 seconds between steps
            n_intervals=0,
            disabled=True
        )
    ]