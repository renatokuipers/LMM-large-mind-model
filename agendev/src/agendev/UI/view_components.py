"""
View container components for the AgenDev UI.

This module contains utility functions for creating view container elements
such as editor, terminal, and browser views.
"""
from typing import Union, List, Optional, Dict, Any
from dash import html, dcc

def create_terminal_view(
    content: str, 
    timestamp: Optional[str] = None,
    command: Optional[str] = None,
    highlight_lines: Optional[List[int]] = None
) -> html.Div:
    """
    Create an enhanced terminal view component with timestamp and optional command highlighting.
    
    Args:
        content: Terminal output text
        timestamp: Optional timestamp for the terminal output
        command: Optional command that generated the output
        highlight_lines: Optional list of line numbers to highlight
        
    Returns:
        Dash component for terminal view
    """
    # Process content to highlight specific lines if requested
    if highlight_lines:
        lines = content.split('\n')
        highlighted_content = []
        for i, line in enumerate(lines):
            if i+1 in highlight_lines:
                highlighted_content.append(
                    html.Div(
                        line,
                        style={
                            "backgroundColor": "rgba(97, 218, 251, 0.2)",
                            "padding": "0 5px",
                            "borderLeft": "3px solid #61dafb"
                        }
                    )
                )
            else:
                highlighted_content.append(line)
        
        terminal_content = html.Div(highlighted_content)
    else:
        terminal_content = html.Pre(content)
    
    children = [terminal_content]
    
    # Add command if provided
    if command:
        children.insert(0, html.Div(
            className="terminal-command",
            style={
                "backgroundColor": "#333",
                "padding": "5px 10px",
                "borderBottom": "1px solid #444",
                "fontWeight": "bold",
                "color": "#00ff00"
            },
            children=[
                html.Span("$ ", style={"color": "#61dafb"}),
                command
            ]
        ))
    
    # Add timestamp if provided
    if timestamp:
        children.append(html.Div(
            className="terminal-timestamp",
            style={
                "padding": "5px 10px",
                "borderTop": "1px solid #444",
                "fontSize": "0.8em",
                "color": "#888",
                "textAlign": "right"
            },
            children=timestamp
        ))
    
    return html.Div(
        className="terminal-view",
        children=children
    )

def create_editor_view(
    filename: str, 
    content: str, 
    language: str = "text",
    diff_mode: bool = False,
    old_content: Optional[str] = None,
    timestamp: Optional[str] = None,
    highlight_lines: Optional[List[int]] = None
) -> html.Div:
    """
    Create an enhanced editor view component with diff support and line highlighting.
    
    Args:
        filename: Name of the file being edited
        content: File content
        language: Programming language for syntax highlighting
        diff_mode: Whether to show diff view
        old_content: Previous content for diff view
        timestamp: Optional timestamp for the editor view
        highlight_lines: Optional list of line numbers to highlight
        
    Returns:
        Dash component for editor view
    """
    # Process content to highlight specific lines if requested
    editor_content = []
    
    if diff_mode and old_content:
        # Create a simple diff view
        old_lines = old_content.split('\n')
        new_lines = content.split('\n')
        
        # Use difflib to compute the differences
        import difflib
        diff = difflib.unified_diff(old_lines, new_lines, lineterm='')
        
        # Format the diff output
        for line in diff:
            if line.startswith('+'):
                # Added line
                editor_content.append(html.Div(
                    line,
                    style={
                        "backgroundColor": "rgba(0, 255, 0, 0.1)",
                        "color": "#00ff00"
                    }
                ))
            elif line.startswith('-'):
                # Removed line
                editor_content.append(html.Div(
                    line,
                    style={
                        "backgroundColor": "rgba(255, 0, 0, 0.1)",
                        "color": "#ff6666"
                    }
                ))
            elif line.startswith('@@'):
                # Section header
                editor_content.append(html.Div(
                    line,
                    style={
                        "backgroundColor": "rgba(97, 218, 251, 0.1)",
                        "color": "#61dafb",
                        "borderBottom": "1px solid #444",
                        "borderTop": "1px solid #444",
                        "padding": "2px 0"
                    }
                ))
            else:
                # Context line
                editor_content.append(line)
        
        editor_content = html.Div(editor_content)
    else:
        # Regular view with optional line highlighting
        if highlight_lines:
            lines = content.split('\n')
            line_elements = []
            for i, line in enumerate(lines):
                if i+1 in highlight_lines:
                    line_elements.append(html.Div(
                        line,
                        style={
                            "backgroundColor": "rgba(97, 218, 251, 0.2)",
                            "padding": "0 5px",
                            "borderLeft": "3px solid #61dafb"
                        }
                    ))
                else:
                    line_elements.append(line)
            
            editor_content = html.Div(line_elements)
        else:
            editor_content = html.Pre(
                content,
                style={"whiteSpace": "pre-wrap"}
            )
    
    children = [
        html.Div(
            className="editor-header",
            children=[
                html.Div(filename),
                html.Div([
                    html.Button(
                        "Diff", 
                        id="diff-button",
                        className="btn-control" + (" active" if diff_mode else "")
                    ),
                    html.Button(
                        "Original", 
                        id="original-button",
                        className="btn-control"
                    ),
                    html.Button(
                        "Modified", 
                        id="modified-button",
                        className="btn-control",
                        style={"color": "#fff" if not diff_mode else "inherit"}
                    ),
                ])
            ]
        ),
        html.Div(
            className="editor-content",
            children=editor_content
        )
    ]
    
    # Add timestamp if provided
    if timestamp:
        children.append(html.Div(
            className="editor-timestamp",
            style={
                "padding": "5px 10px",
                "borderTop": "1px solid #444",
                "fontSize": "0.8em",
                "color": "#888",
                "textAlign": "right"
            },
            children=timestamp
        ))
    
    return html.Div(
        className="editor-view",
        children=children
    )

def create_browser_view(
    url: str = "about:blank",
    content: Optional[str] = None,
    timestamp: Optional[str] = None
) -> html.Div:
    """
    Create an enhanced browser view component with optional content rendering.
    
    Args:
        url: URL to display in the browser frame
        content: Optional HTML content to render instead of URL
        timestamp: Optional timestamp for the browser view
        
    Returns:
        Dash component for browser view
    """
    children = []
    
    # Create browser header with URL bar
    children.append(html.Div(
        className="browser-header",
        style={
            "padding": "5px 10px",
            "backgroundColor": "#2d2d2d",
            "borderBottom": "1px solid #444",
            "display": "flex",
            "alignItems": "center"
        },
        children=[
            html.I(className="fas fa-globe", style={"marginRight": "10px", "color": "#61dafb"}),
            html.Div(
                className="browser-url-bar",
                style={
                    "flex": "1",
                    "backgroundColor": "#1e1e1e",
                    "padding": "5px 10px",
                    "borderRadius": "4px",
                    "color": "#ccc",
                    "fontFamily": "'Consolas', 'Courier New', monospace",
                    "fontSize": "0.9em",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap"
                },
                children=url
            )
        ]
    ))
    
    # Add content or iframe
    if content:
        children.append(html.Div(
            className="browser-content",
            style={
                "backgroundColor": "#fff",
                "height": "calc(100% - 40px)"
            },
            children=html.Iframe(
                srcDoc=content,
                style={"width": "100%", "height": "100%", "border": "none"}
            )
        ))
    else:
        children.append(html.Iframe(
            src=url,
            style={"width": "100%", "height": "calc(100% - 40px)", "border": "none"}
        ))
    
    # Add timestamp if provided
    if timestamp:
        children.append(html.Div(
            className="browser-timestamp",
            style={
                "padding": "5px 10px",
                "borderTop": "1px solid #444",
                "fontSize": "0.8em",
                "color": "#888",
                "backgroundColor": "#2d2d2d",
                "textAlign": "right"
            },
            children=timestamp
        ))
    
    return html.Div(
        className="browser-view",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
        children=children
    )

def create_view_header(title: str = "AgenDev's Computer") -> html.Div:
    """
    Create the header for the view container.
    
    Args:
        title: Header title
        
    Returns:
        Dash component for view header
    """
    return html.Div(
        className="view-header",
        children=[
            html.Div(title),
            html.Button(
                html.I(className="fas fa-expand"),
                className="btn-control",
                id="expand-view"
            )
        ]
    )

def create_view_type_indicator(
    view_type: str = "Editor", 
    operation_type: str = "Working on", 
    file_path: str = "",
    is_replay: bool = False,
    step_number: Optional[int] = None,
    total_steps: Optional[int] = None
) -> html.Div:
    """
    Create an enhanced view type indicator component with replay information.
    
    Args:
        view_type: Current view type (Editor, Terminal, Browser)
        operation_type: Current operation being performed
        file_path: Path of the file being operated on
        is_replay: Whether in replay mode
        step_number: Current step number in replay mode
        total_steps: Total number of steps in replay mode
        
    Returns:
        Dash component for view type indicator
    """
    children = [
        html.Span(
            "AgenDev is using", 
            style={"color": "#888", "marginRight": "5px"}
        ),
        html.Span(
            id="current-view-type", 
            children=view_type,
            style={"fontWeight": "bold", "color": "#61dafb"}
        ),
        html.Div(
            id="file-operation-indicator",
            style={
                "marginLeft": "20px",
                "display": "flex",
                "alignItems": "center",
                "color": "#888",
                "fontSize": "0.85em"
            },
            children=[
                html.Span(
                    id="operation-type", 
                    children=operation_type
                ),
                html.Code(
                    id="current-file-path",
                    children=file_path,
                    style={
                        "marginLeft": "5px",
                        "backgroundColor": "transparent",
                        "padding": "0"
                    }
                )
            ]
        )
    ]
    
    # Add replay indicator if in replay mode
    if is_replay and step_number is not None and total_steps is not None:
        children.append(html.Div(
            id="replay-step-indicator",
            style={
                "marginLeft": "auto",
                "backgroundColor": "rgba(97, 218, 251, 0.2)",
                "padding": "2px 8px",
                "borderRadius": "4px",
                "display": "flex",
                "alignItems": "center",
                "color": "#61dafb",
                "fontSize": "0.85em",
                "fontWeight": "bold"
            },
            children=[
                html.I(
                    className="fas fa-film",
                    style={"marginRight": "5px"}
                ),
                f"Step {step_number}/{total_steps}"
            ]
        ))
    
    return html.Div(
        id="view-type-indicator",
        style={
            "padding": "5px 10px",
            "backgroundColor": "#2d2d2d",
            "borderBottom": "1px solid #444",
            "display": "flex",
            "alignItems": "center"
        },
        children=children
    )

def create_playback_controls(
    total_steps: int = 0,
    current_step: int = 0,
    is_playing: bool = False,
    is_live: bool = True
) -> html.Div:
    """
    Create enhanced playback controls for the view container.
    
    Args:
        total_steps: Total number of steps in the playback
        current_step: Current step index
        is_playing: Whether playback is currently active
        is_live: Whether in live mode (vs. replay mode)
        
    Returns:
        Dash component for playback controls
    """
    # Calculate slider value based on current step
    slider_value = 0
    if total_steps > 1:
        slider_value = (current_step / (total_steps - 1)) * 100
    elif total_steps == 1:
        slider_value = 100
    
    # Determine play button icon based on state
    play_icon_class = "fas fa-pause" if is_playing else "fas fa-play"
    
    # Determine live button class based on state
    live_button_class = "btn-control active" if is_live else "btn-control"
    
    return html.Div(
        className="view-controls",
        children=[
            html.Div(
                className="progress-controls",
                children=[
                    html.Button(
                        html.I(className="fas fa-step-backward"),
                        className="btn-control",
                        id="playback-backward",
                        title="Previous step (←)",
                        disabled=is_live,
                        style={"opacity": "0.5" if is_live else "1"}
                    ),
                    html.Button(
                        html.I(id="play-icon", className=play_icon_class),
                        className="btn-control",
                        id="playback-play",
                        title="Play/Pause (Space)",
                        disabled=is_live,
                        style={"opacity": "0.5" if is_live else "1"}
                    ),
                    html.Button(
                        html.I(className="fas fa-step-forward"),
                        className="btn-control",
                        id="playback-forward",
                        title="Next step (→)",
                        disabled=is_live,
                        style={"opacity": "0.5" if is_live else "1"}
                    ),
                    html.Div(
                        id="playback-slider-container",
                        style={
                            "width": "300px", 
                            "marginLeft": "10px", 
                            "marginRight": "10px",
                            "display": "flex",
                            "alignItems": "center",
                            "position": "relative"
                        },
                        children=[
                            dcc.Slider(
                                id="playback-slider",
                                min=0,
                                max=100,
                                step=1,
                                value=slider_value,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                                updatemode='drag'
                            )
                        ]
                    ),
                    html.Button(
                        html.I(className="fas fa-bolt"),
                        className=live_button_class,
                        id="live-button",
                        title="Live mode (L)"
                    ),
                    html.Div(
                        id="playback-speed-control",
                        style={"marginLeft": "10px"},
                        children=[
                            html.Span("Speed:", style={"fontSize": "0.8em", "marginRight": "5px"}),
                            dcc.Dropdown(
                                id="playback-speed",
                                options=[
                                    {"label": "0.5x", "value": 0.5},
                                    {"label": "1x", "value": 1},
                                    {"label": "2x", "value": 2},
                                    {"label": "4x", "value": 4}
                                ],
                                value=1,
                                clearable=False,
                                style={
                                    "width": "70px", 
                                    "backgroundColor": "#2d2d2d",
                                    "color": "#fff",
                                    "border": "none"
                                }
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                className="status-indicator",
                children=[
                    html.Span(
                        html.I(id="task-status-icon", className="fas fa-spinner fa-spin"),
                        id="task-status-tag",
                        className="status-tag in-progress",
                        style={"marginRight": "5px"}
                    ),
                    html.Span(id="current-task-text", children="Setting up project environment"),
                    html.Span(id="task-progress", children="1/5", className="time-indicator"),
                    html.Div(
                        id="replay-indicator",
                        style={
                            "marginLeft": "10px",
                            "display": "inline-flex",
                            "alignItems": "center",
                            "opacity": "1" if not is_live else "0",
                            "transition": "opacity 0.3s ease"
                        },
                        children=[
                            html.I(className="fas fa-film", style={"marginRight": "5px"}),
                            html.Span("REPLAY MODE", style={"fontSize": "0.8em", "fontWeight": "bold", "color": "#61dafb"})
                        ]
                    )
                ]
            )
        ]
    )

def create_timeline_marker(
    position: float, 
    step_type: str, 
    tooltip_text: str, 
    is_current: bool = False
) -> html.Div:
    """
    Create a timeline marker for the playback slider.
    
    Args:
        position: Position on the timeline (0-100)
        step_type: Type of step ("code", "terminal", "browser", "error", etc.)
        tooltip_text: Text to show in the tooltip
        is_current: Whether this is the current step
        
    Returns:
        Dash component for timeline marker
    """
    # Define icons and colors for different step types
    icon_map = {
        "terminal": "fas fa-terminal",
        "code": "fas fa-code",
        "file": "fas fa-file-code",
        "browser": "fas fa-globe",
        "error": "fas fa-exclamation-circle",
        "success": "fas fa-check-circle",
        "task": "fas fa-tasks",
        "setup": "fas fa-cog",
        "default": "fas fa-circle"
    }
    
    color_map = {
        "terminal": "#61dafb",
        "code": "#f8f8f8",
        "file": "#00ff00",
        "browser": "#ff6b6b",
        "error": "#dc3545",
        "success": "#28a745",
        "task": "#9370db",
        "setup": "#ffc107",
        "default": "#888"
    }
    
    icon_class = icon_map.get(step_type, icon_map["default"])
    color = color_map.get(step_type, color_map["default"])
    
    return html.Div(
        className="timeline-marker",
        style={
            "position": "absolute",
            "left": f"{position}%",
            "bottom": "20px",
            "transform": "translateX(-50%)",
            "zIndex": "10",
            "cursor": "pointer"
        },
        title=tooltip_text,
        children=html.I(
            className=icon_class,
            style={
                "color": color,
                "fontSize": "16px" if is_current else "12px",
                "filter": "drop-shadow(0 0 3px rgba(255,255,255,0.5))" if is_current else "none",
                "transition": "all 0.2s ease"
            }
        )
    )

def create_step_preview(step_data: Dict[str, Any], is_current: bool = False) -> html.Div:
    """
    Create a preview component for a step in the timeline.
    
    Args:
        step_data: Step data including type, operation, file path, etc.
        is_current: Whether this is the current step
        
    Returns:
        Dash component for step preview
    """
    # Determine icon based on step type
    step_type = step_data.get("type", "terminal")
    step_type_icon = {
        "terminal": "fas fa-terminal",
        "editor": "fas fa-file-code",
        "browser": "fas fa-globe"
    }.get(step_type, "fas fa-circle")
    
    # Determine color based on step operation
    operation_type = step_data.get("operation_type", "")
    operation_color = {
        "Setting up": "#ffc107",
        "Configuring": "#61dafb",
        "Creating": "#00ff00",
        "Implementing": "#f8f8f8",
        "Saving": "#28a745",
        "Error": "#dc3545"
    }.get(operation_type, "#888")
    
    # Build preview content
    content = [
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "10px"
            },
            children=[
                html.I(
                    className=step_type_icon,
                    style={
                        "marginRight": "10px",
                        "color": operation_color,
                        "fontSize": "18px"
                    }
                ),
                html.Span(
                    operation_type or step_type.capitalize(),
                    style={
                        "fontWeight": "bold",
                        "color": operation_color
                    }
                )
            ]
        ),
        html.Div(
            style={
                "marginBottom": "5px"
            },
            children=[
                html.Span(
                    "File: ",
                    style={
                        "color": "#888",
                        "marginRight": "5px"
                    }
                ),
                html.Code(
                    step_data.get("file_path", "") or "N/A",
                    style={
                        "backgroundColor": "rgba(97, 218, 251, 0.1)",
                        "padding": "2px 4px",
                        "borderRadius": "2px"
                    }
                )
            ]
        )
    ]
    
    # Add timestamp if available
    if "timestamp" in step_data:
        import datetime
        timestamp = datetime.datetime.fromtimestamp(step_data["timestamp"]).strftime("%H:%M:%S")
        content.append(
            html.Div(
                timestamp,
                style={
                    "color": "#888",
                    "fontSize": "0.8em",
                    "marginTop": "5px"
                }
            )
        )
    
    return html.Div(
        className="step-preview",
        style={
            "backgroundColor": "#2d2d2d" if is_current else "#1e1e1e",
            "border": f"1px solid {operation_color}" if is_current else "1px solid #444",
            "borderRadius": "4px",
            "padding": "10px",
            "transition": "all 0.3s ease",
            "opacity": "1" if is_current else "0.7"
        },
        children=content
    )

def create_keyboard_listener() -> html.Div:
    """
    Create a keyboard event listener for playback navigation.
    
    Returns:
        Keyboard listener component
    """
    return html.Div([
        # Hidden div that listens for keyboard events
        html.Div(id='keyboard-listener', 
                 style={'display': 'none'}),
        
        # Store for keyboard actions and events
        dcc.Store(id='keyboard-action', data=None),
        dcc.Store(id='keyboard-events', data={'n_keydowns': 0, 'keydowns': []}),
        
        # Clientside JavaScript to capture keyboard events
        html.Script('''
            document.addEventListener('keydown', function(e) {
                // Only handle playback control keys
                if (['ArrowLeft', 'ArrowRight', ' ', 'l', 'L', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].includes(e.key)) {
                    // Don't capture when typing in text fields
                    if (document.activeElement.tagName === 'INPUT' || 
                        document.activeElement.tagName === 'TEXTAREA' ||
                        document.activeElement.isContentEditable) {
                        return;
                    }
                    
                    // Trigger a custom event on the keyboard listener
                    var keyEvent = new CustomEvent('dash_keydown', {
                        detail: {
                            key: e.key,
                            timeStamp: new Date().getTime()
                        }
                    });
                    document.getElementById('keyboard-listener').dispatchEvent(keyEvent);
                    
                    // Prevent default behavior for space (page scroll)
                    if (e.key === ' ') {
                        e.preventDefault();
                    }
                }
            });
        ''')
    ])