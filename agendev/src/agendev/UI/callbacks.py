"""
Callback functions for the AgenDev UI with pattern-matching support.

This module contains all Dash callback functions to handle user interactions and UI updates,
utilizing Dash's pattern-matching callbacks for dynamic content.
"""
from __future__ import annotations
from dash import Input, Output, State, ALL, MATCH, callback_context, html, dcc, no_update
from dash.exceptions import PreventUpdate
from typing import Dict, List, Any, Tuple, Union, Optional, cast
import json
import time
from datetime import datetime
import uuid
import logging
from pydantic import BaseModel, Field

from .chat_components import render_markdown, create_command_element, create_file_operation
from .view_components import create_terminal_view, create_editor_view
from .core_integration import CoreIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core integration
core_integration = CoreIntegration(
    llm_base_url="http://192.168.2.12:1234",
    tts_base_url="http://127.0.0.1:7860"
)

# Define a DashComponent type for better type hinting
DashComponent = Any  # This is safer than trying to reference a non-existent base class

# Type definitions for better type safety
class CallbackResult(BaseModel):
    """Base model for callback results to ensure type safety."""
    pass

class TaskExecutionResult(CallbackResult):
    """Result of a task execution operation."""
    task_data: Dict[str, Any] = Field(...)
    todo_data: Dict[str, Any] = Field(...)
    playback_data: Dict[str, Any] = Field(...)

class ViewUpdateResult(CallbackResult):
    """Result of a view content update."""
    content: List[DashComponent] = Field(...)
    view_type: str = Field(...)
    operation_type: str = Field(...)
    file_path: str = Field(...)
    slider_value: float = Field(...)

def register_callbacks(app) -> None:
    """
    Register all callback functions with the Dash app.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [Output("app-state", "data"),
         Output("landing-page", "style"),
         Output("main-container", "style"),
         Output("project-title", "children"),
         Output("todo-data", "data"),
         Output("playback-data", "data"),
         Output("task-data", "data")],
        [Input("submit-button", "n_clicks")],
        [State("initial-prompt", "value"),
         State("app-state", "data"),
         State("todo-data", "data"),
         State("playback-data", "data"),
         State("task-data", "data")],
        prevent_initial_call=True
    )
    def transition_to_main_view(
        n_clicks: int, 
        prompt_value: str, 
        current_state: Dict[str, Any], 
        todo_data: Dict[str, Any], 
        playback_data: Dict[str, Any],
        task_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Transition from landing page to main view when the user submits their project idea.
        
        Args:
            n_clicks: Number of button clicks
            prompt_value: User's project description
            current_state: Current application state
            todo_data: Current todo.md data
            playback_data: Current playback data
            task_data: Current task data
            
        Returns:
            Updated state values
        """
        if not n_clicks:
            raise PreventUpdate
        
        # Clean up the prompt value
        if not prompt_value:
            prompt_value = "New project"
        
        # Extract a title from the prompt
        title = "New Project"
        if len(prompt_value.split()) < 5:
            title = prompt_value.capitalize()
        else:
            words = prompt_value.split()
            title_words = words[:3]
            title = " ".join(word.capitalize() for word in title_words) + " Project"
        
        # Update state
        current_state["view"] = "main"
        current_state["initial_prompt"] = prompt_value
        current_state["project_name"] = title
        
        # Initialize project using core integration
        initialization_result = core_integration.initialize_project(title, prompt_value)
        
        # Generate task data
        tasks = core_integration.get_tasks()
        updated_task_data = {"tasks": []}
        
        for task in tasks:
            task_content = [
                html.P(task["description"]),
                create_command_element(f"cd task/{task['id']}", "completed")
            ]
            
            updated_task_data["tasks"].append({
                "id": task["id"],
                "title": task["title"],
                "status": task.get("status", "planned"),
                "content": task_content
            })
        
        # Generate todo.md content based on tasks
        todo_content = core_integration.generate_todo_markdown(title, tasks)
        todo_data["content"] = todo_content
        
        # Create initial playback steps (setup environment)
        playback_data["steps"] = [
            {
                "type": "terminal",
                "content": f"$ echo 'Initializing {title}'\nInitializing {title}\n$ mkdir {title.lower().replace(' ', '_')}\n$ cd {title.lower().replace(' ', '_')}",
                "operation_type": "Setting up",
                "file_path": title.lower().replace(' ', '_')
            },
            {
                "type": "terminal",
                "content": f"$ echo 'Creating project structure'\nCreating project structure\n$ mkdir -p src tests docs\n$ touch README.md\n$ echo '# {title}' > README.md\n$ echo 'Project setup complete!'",
                "operation_type": "Configuring",
                "file_path": "project structure"
            },
            {
                "type": "editor",
                "filename": "README.md",
                "content": f"# {title}\n\nThis project was created with AgenDev, an Intelligent Agentic Development System.\n\n## Description\n\n{prompt_value}\n\n## Getting Started\n\n```bash\n# Clone or download the project\ncd {title.lower().replace(' ', '-')}\n```\n\n## Features\n\n- Feature 1 (Coming soon)\n- Feature 2 (Coming soon)\n- Feature 3 (Coming soon)\n\n## License\n\nMIT\n",
                "operation_type": "Creating",
                "file_path": "README.md"
            }
        ]
        
        playback_data["total_steps"] = len(playback_data["steps"])
        playback_data["current_step"] = len(playback_data["steps"]) - 1  # Point to latest step
        
        # Hide landing page, show main container
        landing_style = {"display": "none"}
        main_style = {"display": "flex"}
        
        return current_state, landing_style, main_style, title, todo_data, playback_data, updated_task_data

    @app.callback(
        Output("todo-content", "children"),
        [Input("todo-data", "data")],
        prevent_initial_call=True
    )
    def update_todo_content(todo_data: Dict[str, Any]) -> html.Component:
        """
        Update the todo.md content when the data changes.
        
        Args:
            todo_data: Todo data store
            
        Returns:
            Rendered markdown component
        """
        if not todo_data or "content" not in todo_data:
            return render_markdown("# No tasks available")
            
        return render_markdown(todo_data["content"])

    @app.callback(
        Output("task-sections", "children"),
        [Input("task-data", "data")],
        prevent_initial_call=True
    )
    def update_task_sections(task_data: Dict[str, Any]) -> List[html.Div]:
        """
        Update task sections when the task data changes.
        
        Args:
            task_data: Task data store
            
        Returns:
            List of task section components
        """
        task_sections = []
        
        for task in task_data.get("tasks", []):
            # Process task content
            content = []
            if isinstance(task.get("content"), list):
                if task.get("content"):
                    content = task["content"]
                else:
                    # Default content if empty
                    content = [
                        html.P(f"Working on: {task['title']}"),
                        create_command_element("cd feature/task", "completed")
                    ]
            else:
                # If content is a string, convert to paragraph
                content = [html.P(task.get("content", ""))]
            
            # Determine icon class based on status
            if task.get("status") == "completed":
                icon_class = "fas fa-check-circle"
                icon_style = {"marginRight": "10px", "color": "#00ff00"}
            elif task.get("status") == "in-progress":
                icon_class = "fas fa-spinner fa-spin"
                icon_style = {"marginRight": "10px", "color": "#ffc107"}
            elif task.get("status") == "failed":
                icon_class = "fas fa-times-circle"
                icon_style = {"marginRight": "10px", "color": "#dc3545"}
            else:
                icon_class = "fas fa-circle"
                icon_style = {"marginRight": "10px", "color": "#888"}
            
            # Create header with icon and title
            header_content = html.Div([
                html.I(className=icon_class, style=icon_style),
                html.Span(task.get("title", "Unnamed Task"))
            ])
            
            # Create section with pattern-matching IDs
            section = html.Div([
                html.Div(
                    className="collapsible-header",
                    id={"type": "task-header", "index": task["id"]},
                    children=[
                        html.I(
                            className="fas fa-chevron-down mr-2",
                            style={"marginRight": "10px"}
                        ),
                        header_content
                    ]
                ),
                html.Div(
                    id={"type": "task-content", "index": task["id"]},
                    className="collapsible-content",
                    style={"display": "block" if task.get("status") in ["in-progress", "failed"] else "none"},
                    children=content
                )
            ])
            
            task_sections.append(section)
        
        return task_sections

    # Pattern-matching callback for ALL task section toggles
    @app.callback(
        Output({"type": "task-content", "index": ALL}, "style"),
        Input({"type": "task-header", "index": ALL}, "n_clicks"),
        State({"type": "task-content", "index": ALL}, "style"),
        prevent_initial_call=True
    )
    def toggle_task_sections(n_clicks_list: List[int], styles_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Toggle visibility of task sections when headers are clicked.
        Uses pattern-matching to handle any number of sections dynamically.
        
        Args:
            n_clicks_list: List of click counts for all headers
            styles_list: List of current styles for all content sections
            
        Returns:
            Updated list of styles for all content sections
        """
        ctx = callback_context
        if not ctx.triggered:
            return [no_update] * len(styles_list)
        
        # Get the triggered component's ID
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if '{' not in triggered_id:
            return [no_update] * len(styles_list)
        
        # Parse the JSON ID to get the index
        try:
            # Handle both dictionary and string cases
            triggered_dict = json.loads(triggered_id) if isinstance(triggered_id, str) else triggered_id
            triggered_index = triggered_dict['index']
        except (json.JSONDecodeError, KeyError):
            return [no_update] * len(styles_list)
        
        # Create a list of updated styles
        updated_styles = []
        for i, (item_id, style) in enumerate(zip(ctx.inputs_list[0], styles_list)):
            try:
                # Access the dictionary directly - no json.loads() needed
                item_dict = item_id['id']
                item_index = item_dict['index']
                
                # Only update the style for the triggered item
                if item_index == triggered_index:
                    is_visible = style.get('display') == 'block'
                    updated_styles.append({'display': 'none' if is_visible else 'block'})
                else:
                    updated_styles.append(no_update)
            except KeyError:
                updated_styles.append(no_update)
        
        return updated_styles

    @app.callback(
        [Output("view-content", "children"),
         Output("current-view-type", "children"),
         Output("operation-type", "children"),
         Output("current-file-path", "children"),
         Output("playback-slider", "value", allow_duplicate=True)],
        [Input("playback-data", "data")],
        [State("playback-slider", "value")],
        prevent_initial_call=True
    )
    def update_view_content(
        playback_data: Dict[str, Any],
        current_slider_value: float
    ) -> Tuple[List[DashComponent], str, str, str, float]:
        """
        Update the view content based on playback data.
        
        Args:
            playback_data: Playback data store
            current_slider_value: Current slider value
            
        Returns:
            Tuple of (view content, view type, operation type, file path, slider value)
        """
        if not playback_data or not playback_data.get("steps"):
            return [], "None", "", "", 0
        
        current_step = playback_data["current_step"]
        total_steps = playback_data.get("total_steps", len(playback_data["steps"]))
        
        if current_step >= len(playback_data["steps"]):
            current_step = len(playback_data["steps"]) - 1
        
        if current_step < 0:
            current_step = 0
            
        step_data = playback_data["steps"][current_step]
        view_type = step_data.get("type", "terminal")
        
        if view_type == "terminal":
            content = create_terminal_view(step_data.get("content", ""))
        elif view_type == "editor":
            content = create_editor_view(
                step_data.get("filename", "unnamed.txt"),
                step_data.get("content", ""),
                step_data.get("language", "text")
            )
        elif view_type == "browser":
            content = html.Iframe(
                src=step_data.get("url", "about:blank"),
                style={"width": "100%", "height": "100%", "border": "none"}
            )
        else:
            content = html.Div("No content available")
        
        # Update operation indicators
        operation_type = step_data.get("operation_type", "Working on")
        file_path = step_data.get("file_path", "")
        
        # Calculate slider value
        if total_steps > 1:
            slider_value = (current_step / (total_steps - 1)) * 100
        else:
            slider_value = 100
            
        return [content], view_type.capitalize(), operation_type, file_path, slider_value

    @app.callback(
        [Output("playback-data", "data", allow_duplicate=True),
         Output("playback-interval", "disabled"),
         Output("play-icon", "className"),
         Output("live-button", "className"),
         Output("task-status-tag", "className"),
         Output("task-status-icon", "className"),
         Output("current-task-text", "children"),
         Output("task-progress", "children")],
        [Input("playback-backward", "n_clicks"),
         Input("playback-play", "n_clicks"),
         Input("playback-forward", "n_clicks"),
         Input("live-button", "n_clicks"),
         Input("playback-interval", "n_intervals"),
         Input("playback-slider", "value")],
        [State("playback-data", "data"),
         State("app-state", "data")],
        prevent_initial_call=True
    )
    def control_playback(
        backward_clicks: int, 
        play_clicks: int, 
        forward_clicks: int, 
        live_clicks: int, 
        interval: int, 
        slider_value: float, 
        playback_data: Dict[str, Any], 
        app_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool, str, str, str, str, str, str]:
        """
        Control playback based on user interaction with playback controls.
        
        Args:
            backward_clicks: Number of backward button clicks
            play_clicks: Number of play button clicks
            forward_clicks: Number of forward button clicks
            live_clicks: Number of live button clicks
            interval: Number of interval ticks
            slider_value: Current slider value
            playback_data: Current playback data
            app_state: Current application state
            
        Returns:
            Updated playback state and UI indicators
        """
        # Get the component that triggered the callback
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        if not playback_data or not playback_data.get("steps"):
            raise PreventUpdate
        
        # Initialize values
        current_step = playback_data["current_step"]
        is_playing = playback_data.get("is_playing", False)
        is_live = app_state.get("is_live_mode", True)
        total_steps = playback_data.get("total_steps", len(playback_data["steps"]))
        
        # Handle different triggers
        if trigger_id == "playback-backward":
            current_step = max(0, current_step - 1)
            is_playing = False
            is_live = False
        
        elif trigger_id == "playback-play":
            is_playing = not is_playing
            is_live = False
        
        elif trigger_id == "playback-forward":
            current_step = min(total_steps - 1, current_step + 1)
            if current_step == total_steps - 1:
                is_playing = False
                is_live = True
        
        elif trigger_id == "live-button":
            is_live = True
            current_step = total_steps - 1
            is_playing = False
        
        elif trigger_id == "playback-interval" and is_playing:
            current_step = min(total_steps - 1, current_step + 1)
            if current_step == total_steps - 1:
                is_playing = False
                is_live = True
        
        elif trigger_id == "playback-slider":
            # Calculate the step based on slider value
            if total_steps > 1:
                current_step = min(total_steps - 1, max(0, round((slider_value / 100) * (total_steps - 1))))
            else:
                current_step = 0
                
            is_playing = False
            is_live = (current_step == total_steps - 1)
        
        # Update playback data
        playback_data["current_step"] = current_step
        playback_data["is_playing"] = is_playing
        app_state["is_live_mode"] = is_live
        
        # Update task status
        status_class = "status-tag in-progress"
        icon_class = "fas fa-spinner fa-spin"
        
        if is_live and total_steps > 0:
            current_task = "Current task in progress..."
        else:
            step_index = min(current_step, total_steps - 1) if total_steps > 0 else 0
            step_data = playback_data["steps"][step_index] if playback_data["steps"] else {}
            current_task = f"Step {step_index + 1}: {step_data.get('operation_type', '')} {step_data.get('file_path', '')}"
        
        # Task progress indicator
        progress_text = f"{current_step + 1}/{total_steps}"
        
        # Button classes
        play_icon_class = "fas fa-pause" if is_playing else "fas fa-play"
        live_button_class = "btn-control active" if is_live else "btn-control"
        
        return (
            playback_data, 
            not is_playing,  # Interval is disabled when not playing
            play_icon_class,
            live_button_class,
            status_class,
            icon_class,
            current_task,
            progress_text
        )
    
    @app.callback(
        [Output("task-data", "data", allow_duplicate=True),
         Output("todo-data", "data", allow_duplicate=True),
         Output("playback-data", "data", allow_duplicate=True)],
        [Input("execute-task-button", "n_clicks")],
        [State("task-selector", "value"),
         State("app-state", "data"),
         State("todo-data", "data"),
         State("task-data", "data"),
         State("playback-data", "data")],
        prevent_initial_call=True
    )
    def execute_task(
        n_clicks: int,
        task_id: str,
        app_state: Dict[str, Any],
        todo_data: Dict[str, Any],
        task_data: Dict[str, Any],
        playback_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Execute a task and update the UI accordingly.
        
        Args:
            n_clicks: Number of button clicks
            task_id: ID of the task to execute
            app_state: Current application state
            todo_data: Current todo data
            task_data: Current task data
            playback_data: Current playback data
            
        Returns:
            Updated task data, todo data, and playback data
        """
        if not n_clicks or not task_id:
            raise PreventUpdate
            
        logger.info(f"Starting task execution with task_id: {task_id}")
        
        # Update task status to in-progress before starting execution
        updated_tasks = []
        for task in task_data.get("tasks", []):
            if task["id"] == task_id:
                # Update task to show it's in progress
                task["status"] = "in-progress"
                task["content"] = [
                    html.P(task.get("description", "Executing task...")),
                    html.Div([
                        html.I(className="fas fa-spinner fa-spin", 
                              style={"marginRight": "10px", "color": "#ffc107"}),
                        html.Span("Task execution in progress...")
                    ]),
                    create_command_element(f"cd task/{task['id']}", "in-progress")
                ]
            updated_tasks.append(task)
            
        task_data["tasks"] = updated_tasks
        
        # Add a global timeout for the entire task execution
        import threading
        import time
        
        execution_result = {
            "success": False,
            "error": "Task execution timed out"
        }
        execution_completed = False
        
        def execute_task_with_timeout():
            nonlocal execution_result, execution_completed
            try:
                # Execute the task using core integration
                result = core_integration.execute_task(task_id)
                execution_result = result
                logger.info(f"Task execution completed with result: {result}")
            except Exception as e:
                logger.error(f"Error during task execution: {e}")
                import traceback
                traceback.print_exc()
                execution_result = {
                    "success": False,
                    "error": f"Task execution failed: {str(e)}"
                }
            finally:
                execution_completed = True
        
        # Start task execution in a separate thread
        execution_thread = threading.Thread(target=execute_task_with_timeout)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Wait for completion with timeout (60 seconds)
        timeout = 60  # seconds
        start_time = time.time()
        while not execution_completed and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not execution_completed:
            logger.warning(f"Task execution timed out after {timeout} seconds")
            execution_result = {
                "success": False,
                "error": f"Task execution timed out after {timeout} seconds",
                "error_category": "timeout"
            }
        
        # Handle failed task execution gracefully in the UI
        if not execution_result.get("success", False):
            error_message = execution_result.get("error", "Unknown error during task execution")
            error_category = execution_result.get("error_category", "unknown")
            logger.error(f"Task execution failed: {error_message}")
            
            # Update task to show error
            updated_tasks = []
            for task in task_data.get("tasks", []):
                if task["id"] == task_id:
                    task["status"] = "failed"
                    
                    # Create content with appropriate error information
                    error_content = [
                        html.P(task.get("description", "Task failed.")),
                        html.Div([
                            html.I(className="fas fa-times-circle", 
                                style={"marginRight": "10px", "color": "#dc3545"}),
                            html.Span(f"Task execution failed", 
                                    style={"fontWeight": "bold", "color": "#dc3545"})
                        ]),
                        html.P(f"Error: {error_message}", style={"color": "#dc3545"}),
                        html.P(f"Category: {error_category}", style={"color": "#dc3545", "fontSize": "0.9em"}),
                        create_command_element(f"cd task/{task['id']}", "failed")
                    ]
                    
                    # Add retry button for recovery
                    retry_button = html.Button(
                        "Retry Task", 
                        id={"type": "retry-task-button", "index": task_id},
                        style={
                            "marginTop": "10px",
                            "padding": "5px 10px",
                            "backgroundColor": "#61dafb",
                            "color": "black",
                            "border": "none",
                            "borderRadius": "4px",
                            "cursor": "pointer"
                        }
                    )
                    
                    error_content.append(retry_button)
                    task["content"] = error_content
                
                updated_tasks.append(task)
                
            task_data["tasks"] = updated_tasks
            
            # Generate playback steps for the failed task
            new_steps = [{
                "type": "terminal",
                "content": f"$ echo 'Task execution failed'\nError: {error_message}\nCategory: {error_category}",
                "operation_type": "Error",
                "file_path": "Task execution"
            }]
            
            # Append error step to existing playback data
            current_steps = playback_data.get("steps", [])
            playback_data["steps"] = current_steps + new_steps
            playback_data["total_steps"] = len(playback_data["steps"])
            playback_data["current_step"] = len(playback_data["steps"]) - 1  # Point to error step
            
            # Update todo.md to reflect task status
            tasks = core_integration.get_tasks()  # Get fresh task data
            todo_data["content"] = core_integration.generate_todo_markdown(
                app_state.get("project_name", "Project"),
                tasks
            )
            
            return task_data, todo_data, playback_data
            
        # If execution was successful, continue with normal flow
        # Update task status in task data
        updated_tasks = []
        for task in task_data.get("tasks", []):
            if task["id"] == task_id:
                # Update task status and content
                task["status"] = "completed"
                
                # Add completion information to task content
                file_path = execution_result.get("file_path", "")
                task["content"] = [
                    html.P(task.get("description", "Task completed successfully.")),
                    html.Div([
                        html.I(className="fas fa-check-circle", 
                              style={"marginRight": "10px", "color": "#00ff00"}),
                        html.Span("Task completed successfully!")
                    ]),
                    create_command_element(f"cd task/{task['id']}", "completed"),
                    create_file_operation("Created", file_path, "completed")
                ]
            
            updated_tasks.append(task)
            
        task_data["tasks"] = updated_tasks
        
        # Update todo.md to reflect task completion
        tasks = core_integration.get_tasks()  # Get fresh task data
        todo_data["content"] = core_integration.generate_todo_markdown(
            app_state.get("project_name", "Project"),
            tasks
        )
        
        # Generate playback steps for the task execution
        new_steps = core_integration.generate_playback_steps(execution_result)
        
        # Append new steps to existing playback data
        current_steps = playback_data.get("steps", [])
        playback_data["steps"] = current_steps + new_steps
        playback_data["total_steps"] = len(playback_data["steps"])
        playback_data["current_step"] = len(playback_data["steps"]) - 1  # Point to latest step
        
        # Create a validated result using our Pydantic model
        result = TaskExecutionResult(
            task_data=task_data,
            todo_data=todo_data,
            playback_data=playback_data
        )
        
        return result.task_data, result.todo_data, result.playback_data
    
    @app.callback(
        [Output("task-data", "data", allow_duplicate=True),
         Output("todo-data", "data", allow_duplicate=True),
         Output("playback-data", "data", allow_duplicate=True)],
        [Input({"type": "retry-task-button", "index": ALL}, "n_clicks")],
        [State({"type": "retry-task-button", "index": ALL}, "id"),
         State("app-state", "data"),
         State("todo-data", "data"),
         State("task-data", "data"),
         State("playback-data", "data")],
        prevent_initial_call=True
    )
    def retry_failed_task(
        n_clicks_list: List[int],
        button_ids: List[Dict[str, Any]],
        app_state: Dict[str, Any],
        todo_data: Dict[str, Any],
        task_data: Dict[str, Any],
        playback_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Retry a failed task when the retry button is clicked.
        
        Args:
            n_clicks_list: List of button click counts
            button_ids: List of button IDs
            app_state: Current application state
            todo_data: Current todo data
            task_data: Current task data
            playback_data: Current playback data
            
        Returns:
            Updated task data, todo data, and playback data
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if not triggered_id:
            raise PreventUpdate
        
        # Extract the task ID from the button ID
        try:
            button_id = json.loads(triggered_id)
            task_id = button_id["index"]
        except:
            raise PreventUpdate
        
        logger.info(f"Retrying failed task with task_id: {task_id}")
        
        # Reset task status to planned for retry
        updated_tasks = []
        for task in task_data.get("tasks", []):
            if task["id"] == task_id:
                task["status"] = "planned"
                task["content"] = [
                    html.P(task.get("description", "Task ready for retry.")),
                    html.Div([
                        html.I(className="fas fa-redo", 
                            style={"marginRight": "10px", "color": "#61dafb"}),
                        html.Span("Task reset for retry")
                    ]),
                    create_command_element(f"cd task/{task['id']}", "planned")
                ]
            updated_tasks.append(task)
        
        task_data["tasks"] = updated_tasks
        
        # Update todo.md to reflect task status
        tasks = core_integration.get_tasks()  # Get fresh task data
        todo_data["content"] = core_integration.generate_todo_markdown(
            app_state.get("project_name", "Project"),
            tasks
        )
        
        # Add task reset step to playback data
        new_step = {
            "type": "terminal",
            "content": f"$ echo 'Resetting task for retry'\nTask {task_id} reset to planned status for retry.",
            "operation_type": "Retry Preparation",
            "file_path": "Task management"
        }
        
        # Append new step to existing playback data
        current_steps = playback_data.get("steps", [])
        playback_data["steps"] = current_steps + [new_step]
        playback_data["total_steps"] = len(playback_data["steps"])
        playback_data["current_step"] = len(playback_data["steps"]) - 1  # Point to latest step
        
        return task_data, todo_data, playback_data

    @app.callback(
        [Output("task-selector", "options"),
         Output("task-selector", "value")],
        [Input("task-data", "data")],
        prevent_initial_call=True
    )
    def update_task_selector(task_data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        """
        Update the task selector dropdown with available tasks.
        
        Args:
            task_data: Task data store
            
        Returns:
            Task selector options and default value
        """
        options = []
        default_value = ""
        
        for task in task_data.get("tasks", []):
            if task.get("status") not in ["completed", "failed"]:
                options.append({
                    "label": task.get("title", "Unnamed Task"),
                    "value": task.get("id", str(uuid.uuid4()))
                })
                
                # Select the first non-completed task by default
                if not default_value:
                    default_value = task.get("id", "")
        
        return options, default_value
    
    @app.callback(
        [Output("task-data", "data", allow_duplicate=True),
         Output("todo-data", "data", allow_duplicate=True)],
        [Input("refresh-status-button", "n_clicks")],
        [State("app-state", "data"),
         State("task-data", "data"),
         State("todo-data", "data")],
        prevent_initial_call=True
    )
    def refresh_project_status(
        n_clicks: int,
        app_state: Dict[str, Any],
        task_data: Dict[str, Any],
        todo_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Refresh the project status from the core system.
        
        Args:
            n_clicks: Number of button clicks
            app_state: Current application state
            task_data: Current task data
            todo_data: Current todo data
            
        Returns:
            Updated task data and todo data
        """
        if not n_clicks:
            raise PreventUpdate
            
        # Get fresh task data from core integration
        tasks = core_integration.get_tasks()
        
        # Convert to task data format for UI
        updated_task_data = {"tasks": []}
        
        for task in tasks:
            task_content = []
            
            # Add basic information
            task_content.append(html.P(task.get("description", "")))
            
            # Add commands and file operations based on status
            if task.get("status") == "completed":
                task_content.append(
                    html.Div([
                        html.I(className="fas fa-check-circle", 
                            style={"marginRight": "10px", "color": "#00ff00"}),
                        html.Span("Task completed successfully!")
                    ])
                )
                task_content.append(create_command_element(f"cd task/{task['id']}", "completed"))
                
                # Add file operations if available
                for file_path in task.get("artifact_paths", []):
                    task_content.append(create_file_operation("Created", file_path, "completed"))
            elif task.get("status") == "failed":
                task_content.append(
                    html.Div([
                        html.I(className="fas fa-times-circle", 
                            style={"marginRight": "10px", "color": "#dc3545"}),
                        html.Span("Task execution failed", 
                                style={"fontWeight": "bold", "color": "#dc3545"})
                    ])
                )
                task_content.append(create_command_element(f"cd task/{task['id']}", "failed"))
                
                # Add retry button for recovery
                retry_button = html.Button(
                    "Retry Task", 
                    id={"type": "retry-task-button", "index": task['id']},
                    style={
                        "marginTop": "10px",
                        "padding": "5px 10px",
                        "backgroundColor": "#61dafb",
                        "color": "black",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer"
                    }
                )
                task_content.append(retry_button)
            elif task.get("status") == "in_progress":
                task_content.append(
                    html.Div([
                        html.I(className="fas fa-spinner fa-spin", 
                            style={"marginRight": "10px", "color": "#ffc107"}),
                        html.Span("Task execution in progress...")
                    ])
                )
                task_content.append(create_command_element(f"cd task/{task['id']}", "in-progress"))
            else:
                task_content.append(create_command_element(f"cd task/{task['id']}", "planned"))
            
            updated_task_data["tasks"].append({
                "id": task["id"],
                "title": task["title"],
                "status": task.get("status", "planned"),
                "content": task_content
            })
        
        # Update todo.md content
        updated_todo_data = dict(todo_data)
        updated_todo_data["content"] = core_integration.generate_todo_markdown(
            app_state.get("project_name", "Project"),
            tasks
        )
        
        return updated_task_data, updated_todo_data
        
    @app.callback(
        [Output("task-status-tag", "className"),
         Output("task-status-icon", "className"),
         Output("current-task-text", "children")],
        [Input("task-data", "data")],
        prevent_initial_call=True
    )
    def update_task_status_indicators(task_data: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Update task status indicators in the UI based on task status.
        
        Args:
            task_data: Current task data
            
        Returns:
            Updated status tag class, status icon class, and status text
        """
        # Default values
        status_class = "status-tag in-progress"
        icon_class = "fas fa-spinner fa-spin"
        status_text = "Working on tasks..."
        
        # Count tasks by status
        status_counts = {}
        for task in task_data.get("tasks", []):
            status = task.get("status", "planned")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate total and completed tasks
        total_tasks = len(task_data.get("tasks", []))
        completed_tasks = status_counts.get("completed", 0)
        
        # Check for in-progress tasks
        in_progress_tasks = [task for task in task_data.get("tasks", []) 
                            if task.get("status") == "in-progress"]
        if in_progress_tasks:
            # Show the first in-progress task
            current_task = in_progress_tasks[0]
            status_class = "status-tag in-progress"
            icon_class = "fas fa-spinner fa-spin"
            status_text = f"Executing: {current_task.get('title', 'Current task')}"
        elif status_counts.get("failed", 0) > 0:
            # Show failed status if any tasks failed
            status_class = "status-tag error"
            icon_class = "fas fa-times-circle"
            status_text = f"{status_counts.get('failed', 0)} task(s) failed"
        elif completed_tasks == total_tasks and total_tasks > 0:
            # All tasks completed
            status_class = "status-tag success"
            icon_class = "fas fa-check-circle"
            status_text = "All tasks completed successfully"
        else:
            # Default status
            remaining = total_tasks - completed_tasks
            status_class = "status-tag in-progress"
            icon_class = "fas fa-tasks"
            status_text = f"{remaining} task(s) remaining"
        
        return status_class, icon_class, status_text