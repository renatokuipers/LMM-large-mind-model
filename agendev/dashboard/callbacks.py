"""Callbacks for the AgenDev Dashboard."""

import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import json
import time
from datetime import datetime
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agendev.utils.fs_utils import resolve_path
from agendev.models.task_models import TaskStatus, Task, TaskType, TaskPriority, TaskRisk
from autonomous import (
    generate_tasks_from_description, start_autonomous_process, 
    stop_autonomous_process, get_autonomous_status, add_user_response,
    generate_alternative_plan, build_project_context, analyze_project_risk,
    generate_tests_for_task
)
from .components import create_chat_message
from agendev.llm_module import LLMClient

def register_callbacks(app, agendev):
    """Register all callbacks for the dashboard."""
    
    @app.callback(
        Output("project-form-container", "style"),
        Output("project-info-container", "style"),
        Input("interval-component", "n_intervals")
    )
    def toggle_project_sections(n_intervals):
        # Check if project has started (has tasks)
        has_project = len(agendev.task_graph.tasks) > 0
        
        if has_project:
            return {"display": "none"}, {"display": "block"}
        else:
            return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("chat-messages", "children"),
        [Input("chat-history", "data"),
         Input("interval-component", "n_intervals")]
    )
    def update_chat_messages(chat_history, n_intervals):
        """Update chat messages with system status and user interactions."""
        if not chat_history or "messages" not in chat_history:
            # Initialize with a welcome message
            messages = [
                create_chat_message(
                    "AgenDev", 
                    "Welcome to AgenDev! I'm your autonomous development assistant. "
                    "Create a new project or ask me questions about the current project.",
                    datetime.now().strftime("%H:%M")
                )
            ]
        else:
            messages = [create_chat_message(msg["sender"], msg["message"], msg["timestamp"]) 
                      for msg in chat_history["messages"]]
        
        return messages

    @app.callback(
        Output("chat-history", "data"),
        [Input("send-message", "n_clicks"),
         Input("chat-input", "value")],
        [State("chat-history", "data"),
         State("chat-input", "value")]
    )
    def handle_chat_interactions(n_clicks, input_trigger, chat_history, message):
        """Handle user messages and update chat history."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return chat_history
        
        # Initialize chat history if needed
        if not chat_history:
            chat_history = {"messages": []}
        elif "messages" not in chat_history:
            chat_history["messages"] = []
        
        # Handle user sending a message
        if ctx.triggered[0]["prop_id"] == "send-message.n_clicks" and message and n_clicks:
            # Add user message
            chat_history["messages"].append({
                "sender": "User",
                "message": message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # Process the message through AgenDev LLM
            try:
                # Create a prompt for the LLM
                system_message = "You are an AI assistant for the AgenDev system. Answer questions about the current project or general programming inquiries."
                
                # Process through LLM integration
                llm_response = agendev.llm.query(
                    prompt=message,
                    config=None,  # Use default config
                    clear_context=False,
                    save_to_context=True
                )
                
                # Add AI response
                chat_history["messages"].append({
                    "sender": "AgenDev",
                    "message": llm_response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            except Exception as e:
                # Add error response in case of failure
                chat_history["messages"].append({
                    "sender": "AgenDev",
                    "message": f"I encountered an error while processing your request: {str(e)}",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            
            # Clear the input field
            return chat_history
        
        # Return unchanged chat history for other triggers
        return chat_history

    # Additional callback to clear the input field after sending
    @app.callback(
        Output("chat-input", "value"),
        [Input("send-message", "n_clicks")],
        [State("chat-input", "value")]
    )
    def clear_input_after_send(n_clicks, current_value):
        """Clear the input field after sending a message."""
        if n_clicks and current_value:
            return ""
        return current_value

    @app.callback(
        Output("code-viewer", "children"),
        Input("implementation-file", "value"),
        prevent_initial_call=True
    )
    def update_code_viewer(task_id):
        if not task_id:
            return html.P("No file selected")
        
        task = agendev.task_graph.tasks.get(task_id)
        if not task or not task.artifact_paths:
            return html.P("No artifact found for this task")
        
        try:
            file_path = resolve_path(task.artifact_paths[0])
            if not file_path.exists():
                return html.P(f"File not found: {task.artifact_paths[0]}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            return html.Div([
                html.H6(Path(task.artifact_paths[0]).name, className="mb-2"),
                dbc.Card(
                    dbc.CardBody(
                        dcc.Markdown(
                            f"```python\n{content}\n```",
                            className="mb-0",
                            style={"fontSize": "0.85rem"}
                        )
                    ),
                    className="bg-dark"
                )
            ])
        except Exception as e:
            return html.P(f"Error loading file: {e}")

    @app.callback(
        Output("autonomous-status-badge", "children"),
        Output("autonomous-status-badge", "color"),
        Input("interval-component", "n_intervals")
    )
    def update_status_badges(n_intervals):
        is_running = get_autonomous_status()
        
        if is_running:
            return "Active", "success"
        else:
            return "Inactive", "secondary"

    @app.callback(
        Output("project-creation-output", "children"),
        Input("btn-create-project", "n_clicks"),
        State("project-name", "value"),
        State("project-description", "value"),
        State("chat-history", "data"),
        prevent_initial_call=True
    )
    def handle_create_project(n_clicks, project_name, project_description, chat_history):
        if not project_name or not project_description:
            return dbc.Alert("Project name and description are required", color="danger")
        
        try:
            # Update project name
            agendev.config.project_name = project_name
            
            # Generate tasks and epics based on description
            result = generate_tasks_from_description(agendev, project_name, project_description)
            
            if not result.get("success", False):
                return dbc.Alert(f"Error generating project plan: {result.get('error', 'Unknown error')}", color="danger")
            
            # Generate implementation plan
            agendev.generate_implementation_plan()
            
            # Start autonomous process
            start_result = start_autonomous_process(agendev)
            
            # Add creation message to chat history
            if chat_history:
                if "messages" not in chat_history:
                    chat_history["messages"] = []
                    
                chat_history["messages"].append({
                    "sender": "AgenDev",
                    "message": f"Project '{project_name}' created successfully with {result['epic_count']} epics and {result['task_count']} tasks. Starting autonomous development!",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            
            return dbc.Alert(
                f"Project '{project_name}' created successfully with {result['epic_count']} epics and {result['task_count']} tasks.",
                color="success"
            )
        except Exception as e:
            return dbc.Alert(f"Error creating project: {str(e)}", color="danger")

    @app.callback(
        Output("auto-process-output", "children"),
        Input("btn-start-auto", "n_clicks"),
        Input("btn-stop-auto", "n_clicks"),
        State("chat-history", "data"),
        prevent_initial_call=True
    )
    def handle_auto_process(start_clicks, stop_clicks, chat_history):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-start-auto":
            result = start_autonomous_process(agendev)
            
            # Add message to chat
            if chat_history and result.get("success", False):
                if "messages" not in chat_history:
                    chat_history["messages"] = []
                    
                chat_history["messages"].append({
                    "sender": "AgenDev",
                    "message": "Autonomous development process started! I'll implement your project and keep you updated on progress.",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
            color = "success" if result.get("success", False) else "danger"
            return dbc.Alert(result.get("message", ""), color=color)
        
        elif button_id == "btn-stop-auto":
            result = stop_autonomous_process()
            
            # Add message to chat
            if chat_history and result.get("success", False):
                if "messages" not in chat_history:
                    chat_history["messages"] = []
                    
                chat_history["messages"].append({
                    "sender": "AgenDev",
                    "message": "Autonomous development process has been stopped. You can restart it anytime.",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
            color = "success" if result.get("success", False) else "danger"
            return dbc.Alert(result.get("message", ""), color=color)
        
        return ""

    @app.callback(
        Output({"type": "question-result", "index": dash.dependencies.MATCH}, "children"),
        Input({"type": "question-submit", "index": dash.dependencies.MATCH}, "n_clicks"),
        State({"type": "question-response", "index": dash.dependencies.MATCH}, "value"),
        State({"type": "question-submit", "index": dash.dependencies.MATCH}, "id"),
        State("chat-history", "data"),
        prevent_initial_call=True
    )
    def handle_question_response(n_clicks, response, id_data, chat_history):
        if not response:
            return dbc.Alert("Please provide a response", color="danger")
        
        # Get the task ID from the ID
        task_id = id_data["index"]
        
        # Store the response
        add_user_response(task_id, response)
        
        # Add response to chat
        if chat_history:
            chat_history["messages"].append({
                "sender": "User",
                "message": f"Response: {response}",
                "timestamp": datetime.now().strftime("%H:%M"),
                "response_to": task_id
            })
            
            chat_history["messages"].append({
                "sender": "AgenDev",
                "message": "Thank you for your response. I'll continue with the implementation.",
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        return dbc.Alert("Response submitted. The system will continue with implementation.", color="success")

    @app.callback(
        Output("settings-modal", "is_open"),
        Input("settings-button", "n_clicks"),
        Input("settings-save", "n_clicks"),
        State("settings-modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_settings_modal(settings_click, save_click, is_open):
        """Toggle the settings modal."""
        if settings_click or save_click:
            return not is_open
        return is_open

    @app.callback(
        Output("auto-save-interval-container", "style"),
        Input("settings-auto-save", "value")
    )
    def toggle_auto_save_interval(auto_save_enabled):
        """Show/hide the auto-save interval input based on auto-save setting."""
        if auto_save_enabled and 1 in auto_save_enabled:
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("settings-alert", "children"),
        Output("app-settings", "data"),
        Output("interval-component", "interval"),
        Input("settings-save", "n_clicks"),
        State("settings-project-name", "value"),
        State("settings-notifications-enabled", "value"),
        State("settings-refresh-interval", "value"),
        State("settings-llm-url", "value"),
        State("settings-tts-url", "value"),
        State("settings-default-model", "value"),
        State("settings-auto-save", "value"),
        State("settings-auto-save-interval", "value"),
        State("settings-color-theme", "value"),
        State("settings-font-size", "value"),
        State("app-settings", "data"),
        prevent_initial_call=True
    )
    def save_settings(
        n_clicks, project_name, notifications_enabled, refresh_interval, 
        llm_url, tts_url, default_model, auto_save, auto_save_interval,
        color_theme, font_size, current_settings
    ):
        """Save settings and update configurations."""
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Validate settings
        if not project_name:
            project_name = "AgenDev Dashboard"
        
        if not llm_url:
            llm_url = "http://192.168.2.12:1234"
            
        if not tts_url:
            tts_url = "http://127.0.0.1:7860"
            
        if not default_model:
            default_model = "qwen2.5-7b-instruct"
        
        # Update AgenDev config
        try:
            agendev.config.project_name = project_name
            agendev.config.notifications_enabled = bool(notifications_enabled and 1 in notifications_enabled)
            agendev.config.llm_base_url = llm_url
            agendev.config.tts_base_url = tts_url
            agendev.config.default_model = default_model
            agendev.config.auto_save = bool(auto_save and 1 in auto_save)
            agendev.config.auto_save_interval_minutes = float(auto_save_interval)
            
            # Calculate UI refresh interval in milliseconds
            ui_refresh_interval = int(refresh_interval) * 1000  # Convert to milliseconds
            
            # Save to settings store
            settings = {
                "project_name": project_name,
                "notifications_enabled": bool(notifications_enabled and 1 in notifications_enabled),
                "refresh_interval": refresh_interval,
                "llm_url": llm_url,
                "tts_url": tts_url,
                "default_model": default_model,
                "auto_save": bool(auto_save and 1 in auto_save),
                "auto_save_interval": auto_save_interval,
                "color_theme": color_theme,
                "font_size": font_size
            }
            
            # Save settings to a file to make them persistent
            settings_path = resolve_path("settings.json", create_parents=True)
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Apply theme changes (would be implemented in a real system)
            
            return dbc.Alert(
                "Settings saved successfully",
                color="success",
                duration=4000,  # Auto-dismiss after 4 seconds
                dismissable=True
            ), settings, ui_refresh_interval
            
        except Exception as e:
            return dbc.Alert(
                f"Error saving settings: {str(e)}",
                color="danger",
                dismissable=True
            ), current_settings, dash.no_update

    @app.callback(
        Output("settings-project-name", "value"),
        Output("settings-notifications-enabled", "value"),
        Output("settings-refresh-interval", "value"),
        Output("settings-llm-url", "value"),
        Output("settings-tts-url", "value"),
        Output("settings-default-model", "value"),
        Output("settings-auto-save", "value"),
        Output("settings-auto-save-interval", "value"),
        Output("settings-color-theme", "value"),
        Output("settings-font-size", "value"),
        Input("settings-reset", "n_clicks"),
        prevent_initial_call=True
    )
    def reset_settings(n_clicks):
        """Reset settings to default values."""
        if not n_clicks:
            return dash.no_update
        
        return (
            "AgenDev Dashboard",  # project_name
            [1],  # notifications_enabled
            5,    # refresh_interval
            "http://192.168.2.12:1234",  # llm_url
            "http://127.0.0.1:7860",     # tts_url
            "qwen2.5-7b-instruct",       # default_model
            [1],  # auto_save
            5.0,  # auto_save_interval
            "blue-purple",  # color_theme
            "medium"        # font_size
        )

    @app.callback(
        Output("app-settings", "data", allow_duplicate=True),
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True
    )
    def load_settings(n_intervals):
        """Load settings from disk at startup."""
        if n_intervals != 0:
            return dash.no_update
            
        try:
            # Try to load saved settings
            settings_path = resolve_path("settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    
                # Apply settings to the app
                agendev.config.project_name = settings.get("project_name", agendev.config.project_name)
                agendev.config.notifications_enabled = settings.get("notifications_enabled", agendev.config.notifications_enabled)
                agendev.config.llm_base_url = settings.get("llm_url", agendev.config.llm_base_url)
                agendev.config.tts_base_url = settings.get("tts_url", agendev.config.tts_base_url)
                agendev.config.default_model = settings.get("default_model", agendev.config.default_model)
                agendev.config.auto_save = settings.get("auto_save", agendev.config.auto_save)
                agendev.config.auto_save_interval_minutes = settings.get("auto_save_interval", agendev.config.auto_save_interval_minutes)
                
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return {} 

    # Advanced Features Callbacks
    
    @app.callback(
        Output("alternative-plan-results", "children"),
        Input("generate-alternative-plan-button", "n_clicks"),
        State("optimization-goal", "value"),
        prevent_initial_call=True
    )
    def generate_alternative_plan_callback(n_clicks, optimization_goal):
        """Generate an alternative implementation plan using A* algorithm."""
        if not n_clicks:
            return []
        
        try:
            # Try to get a plan - first try get_latest_plan, then fallback to get_current_plan
            current_plan = None
            
            # Now the PlanningHistory class has get_latest_plan implemented
            if hasattr(agendev.planning_history, 'get_latest_plan'):
                current_plan = agendev.planning_history.get_latest_plan()
            
            # Fallback to get_current_plan if get_latest_plan fails or returns None
            if current_plan is None and hasattr(agendev.planning_history, 'get_current_plan'):
                current_plan = agendev.planning_history.get_current_plan()
            
            # If we still don't have a plan, show error message
            if not current_plan:
                return html.Div([
                    html.P("No current plan available. Please generate a plan first.", className="text-warning")
                ])
                
            # Generate alternative plan
            result = generate_alternative_plan(agendev, optimization_goal)
            
            if not result.get("success", False):
                return html.Div([
                    html.P(f"Error generating alternative plan: {result.get('error', 'Unknown error')}", className="text-danger")
                ])
                
            # Create comparison table
            comparison = result.get("comparison", {})
            
            return html.Div([
                html.H5(f"Alternative Plan ({optimization_goal.capitalize()})"),
                html.P(f"Generated a plan with {len(result['task_sequence'])} tasks."),
                
                # Comparison table
                html.Div([
                    html.H6("Plan Comparison"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric"),
                                html.Th("A* Plan"),
                                html.Th("MCTS Plan"),
                                html.Th("Difference")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td("Task Count"),
                                html.Td(comparison.get("astar_task_count", "N/A")),
                                html.Td(comparison.get("mcts_task_count", "N/A")),
                                html.Td(f"{comparison.get('astar_task_count', 0) - comparison.get('mcts_task_count', 0)}")
                            ]),
                            html.Tr([
                                html.Td("Duration (hours)"),
                                html.Td(f"{comparison.get('astar_duration', 'N/A'):.1f}" if isinstance(comparison.get('astar_duration'), (int, float)) else "N/A"),
                                html.Td(f"{comparison.get('mcts_duration', 'N/A'):.1f}" if isinstance(comparison.get('mcts_duration'), (int, float)) else "N/A"),
                                html.Td(f"{comparison.get('duration_diff_percent', 'N/A'):.1f}%" if isinstance(comparison.get('duration_diff_percent'), (int, float)) else "N/A")
                            ])
                        ])
                    ], bordered=True, dark=True, hover=True, responsive=True, size="sm", striped=True)
                ]),
                
                # Task sequence
                html.Div([
                    html.H6("Task Sequence"),
                    html.Ol([
                        html.Li(agendev.task_graph.tasks[task_id].title) 
                        for task_id in [UUID(tid) for tid in result["task_sequence"]]
                        if task_id in agendev.task_graph.tasks
                    ])
                ], className="mt-3")
            ])
        except Exception as e:
            # Log the exception
            import traceback
            print(f"Error in generate_alternative_plan_callback: {str(e)}")
            print(traceback.format_exc())
            
            # Return friendly error message
            return html.Div([
                html.H5("Error Generating Alternative Plan", className="text-danger"),
                html.P(f"An error occurred: {str(e)}"),
                html.Pre(traceback.format_exc(), className="bg-dark p-3 text-white", style={"fontSize": "0.8rem"})
            ])
    
    @app.callback(
        Output("context-results", "children"),
        Input("build-context-button", "n_clicks"),
        prevent_initial_call=True
    )
    def build_context_callback(n_clicks):
        """Build code context for better understanding of the project."""
        if not n_clicks:
            return []
            
        # Build context
        result = build_project_context(agendev)
        
        if not result.get("success", False):
            return html.Div([
                html.P(f"Error building context: {result.get('error', 'Unknown error')}", className="text-danger")
            ])
            
        # Show results
        return html.Div([
            html.H5("Code Context Built"),
            html.P(f"Successfully indexed {result.get('indexed_elements', 0)} code elements."),
            
            # Context visualization
            html.Div([
                html.H6("Top Code Elements"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Type"),
                            html.Th("Name"),
                            html.Th("File")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(element.get("type", "unknown")),
                            html.Td(element.get("name", "unknown")),
                            html.Td(element.get("file", "unknown"))
                        ]) for element in agendev.context_manager.get_top_elements(10)
                    ])
                ], bordered=True, dark=True, hover=True, responsive=True, size="sm", striped=True)
            ], className="mt-3")
        ])
    
    @app.callback(
        Output("risk-analysis-results", "children"),
        Input("analyze-risk-button", "n_clicks"),
        prevent_initial_call=True
    )
    def analyze_risk_callback(n_clicks):
        """Analyze project risk using probability modeling."""
        if not n_clicks:
            return []
            
        try:
            # Try to get a plan - first try get_latest_plan, then fallback to get_current_plan
            current_plan = None
            
            # Now the PlanningHistory class has get_latest_plan implemented
            if hasattr(agendev.planning_history, 'get_latest_plan'):
                current_plan = agendev.planning_history.get_latest_plan()
            
            # Fallback to get_current_plan if get_latest_plan fails or returns None
            if current_plan is None and hasattr(agendev.planning_history, 'get_current_plan'):
                current_plan = agendev.planning_history.get_current_plan()
            
            # If we still don't have a plan, show error message
            if not current_plan:
                return html.Div([
                    html.P("No current plan available. Please generate a plan first.", className="text-warning")
                ])
                
            # Analyze risk
            result = analyze_project_risk(agendev, current_plan)
            
            # Create risk visualization
            success_probability = result.get("success_probability", 0.5)
            risk_hotspots = result.get("risk_hotspots", [])
            simulation_results = result.get("simulation_results", {})
            
            # Create gauge chart for success probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = success_probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Success Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': success_probability * 100
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor = "rgba(0,0,0,0)",
                font = {'color': "white"},
                height = 300
            )
            
            # Handle potential format for risk_hotspots
            formatted_hotspots = []
            if isinstance(risk_hotspots, list):
                for item in risk_hotspots:
                    if isinstance(item, dict) and 'task_id' in item and 'success_probability' in item:
                        # Case when risk_hotspots is a list of dicts with task_id and success_probability
                        task_id = UUID(item['task_id']) if isinstance(item['task_id'], str) else item['task_id']
                        probability = item['success_probability']
                        formatted_hotspots.append((task_id, probability))
                    elif isinstance(item, tuple) and len(item) == 2:
                        # Case when risk_hotspots is a list of tuples (task_id, probability)
                        formatted_hotspots.append(item)
            
            return html.Div([
                html.H5("Project Risk Analysis"),
                
                # Success probability gauge
                dcc.Graph(figure=fig),
                
                # Risk hotspots
                html.Div([
                    html.H6("Risk Hotspots"),
                    html.P("Tasks with highest risk of failure:"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Task"),
                                html.Th("Failure Probability")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(agendev.task_graph.tasks[task_id].title if task_id in agendev.task_graph.tasks else "Unknown"),
                                html.Td(f"{(1 - probability) * 100:.1f}%")
                            ]) for task_id, probability in formatted_hotspots
                        ])
                    ], bordered=True, dark=True, hover=True, responsive=True, size="sm", striped=True)
                ], className="mt-3"),
                
                # Simulation results
                html.Div([
                    html.H6("Monte Carlo Simulation Results"),
                    html.P(f"Mean completion time: {simulation_results.get('mean_completion_time', 'N/A'):.1f} hours" 
                        if isinstance(simulation_results.get('mean_completion_time'), (int, float)) else "Mean completion time: N/A"),
                    html.P(f"Completion probability: {simulation_results.get('completion_probability', 'N/A') * 100:.1f}%" 
                        if isinstance(simulation_results.get('completion_probability'), (int, float)) else "Completion probability: N/A")
                ], className="mt-3")
            ])
        except Exception as e:
            # Log the exception
            import traceback
            print(f"Error in analyze_risk_callback: {str(e)}")
            print(traceback.format_exc())
            
            # Return friendly error message
            return html.Div([
                html.H5("Error in Risk Analysis", className="text-danger"),
                html.P(f"An error occurred: {str(e)}"),
                html.Pre(traceback.format_exc(), className="bg-dark p-3 text-white", style={"fontSize": "0.8rem"})
            ])
    
    @app.callback(
        Output("snapshots-list", "children"),
        Input("refresh-snapshots-button", "n_clicks"),
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True
    )
    def refresh_snapshots_callback(n_clicks, n_intervals):
        """Refresh the list of code snapshots."""
        # Get snapshots
        snapshots = agendev.snapshot_engine.get_all_snapshots()
        
        if not snapshots:
            return html.Div([
                html.P("No snapshots available.", className="text-muted")
            ])
            
        # Create snapshot list
        return html.Div([
            html.H5(f"Code Snapshots ({len(snapshots)})"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("ID"),
                        html.Th("File"),
                        html.Th("Timestamp"),
                        html.Th("Tags"),
                        html.Th("Actions")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(snapshot.snapshot_id[:8] + "..."),
                        html.Td(Path(snapshot.file_path).name),
                        html.Td(snapshot.timestamp.strftime("%Y-%m-%d %H:%M")),
                        html.Td(", ".join(snapshot.tags)),
                        html.Td(
                            dbc.Button(
                                "View", 
                                id={"type": "view-snapshot", "index": i},
                                color="primary",
                                size="sm"
                            )
                        )
                    ]) for i, snapshot in enumerate(snapshots)
                ])
            ], bordered=True, dark=True, hover=True, responsive=True, size="sm", striped=True)
        ])
    
    @app.callback(
        Output("test-results", "children"),
        Input("generate-tests-button", "n_clicks"),
        State("test-path-input", "value"),
        prevent_initial_call=True
    )
    def generate_tests_callback(n_clicks, file_path):
        """Generate tests for a specific file."""
        if not n_clicks or not file_path:
            return []
            
        # Resolve path
        try:
            resolved_path = resolve_path(file_path, agendev.workspace_dir)
            if not resolved_path.exists():
                return html.Div([
                    html.P(f"File not found: {file_path}", className="text-danger")
                ])
                
            # Create a dummy task for test generation
            task_id = UUID('00000000-0000-0000-0000-000000000000')
            task = agendev.task_graph.tasks.get(task_id)
            if not task:
                # Create a temporary task
                task = Task(
                    task_id=task_id,
                    title="Test Generation",
                    description="Generate tests for file",
                    task_type=TaskType.TEST,
                    priority=TaskPriority.MEDIUM,
                    risk=TaskRisk.LOW,
                    status=TaskStatus.IN_PROGRESS,
                    estimated_duration_hours=1.0,
                    artifact_paths=[str(resolved_path)]
                )
                # Temporarily add to task graph
                agendev.task_graph.tasks[task_id] = task
                
            # Generate tests
            result = generate_tests_for_task(agendev, task_id)
            
            # Clean up temporary task if we created one
            if task_id in agendev.task_graph.tasks and task.title == "Test Generation":
                del agendev.task_graph.tasks[task_id]
                
            if not result.get("success", False):
                return html.Div([
                    html.P(f"Error generating tests: {result.get('error', 'Unknown error')}", className="text-danger")
                ])
                
            # Show results
            test_files = result.get("test_files", [])
            
            return html.Div([
                html.H5("Test Generation Results"),
                html.P(f"Generated {len(test_files)} test files:"),
                html.Ul([
                    html.Li(str(test_file)) for test_file in test_files
                ]),
                dbc.Button(
                    "View Test Results", 
                    id="view-test-results-button", 
                    color="primary",
                    className="mt-3"
                ) if test_files else html.Div()
            ])
        except Exception as e:
            return html.Div([
                html.P(f"Error: {str(e)}", className="text-danger")
            ]) 