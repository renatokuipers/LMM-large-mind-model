"""
View container components for the AgenDev UI.

This module contains utility functions for creating view container elements
such as editor, terminal, and browser views.
"""
from typing import Union
from dash import html, dcc

def create_terminal_view(content: str) -> html.Div:
    """
    Create a terminal view component.
    
    Args:
        content: Terminal output text
        
    Returns:
        Dash component for terminal view
    """
    return html.Div(
        className="terminal-view",
        children=[html.Pre(content)]
    )

def create_editor_view(filename: str, content: str, language: str = "text") -> html.Div:
    """
    Create an editor view component with syntax highlighting.
    
    Args:
        filename: Name of the file being edited
        content: File content
        language: Programming language for syntax highlighting
        
    Returns:
        Dash component for editor view
    """
    return html.Div(
        className="editor-view",
        children=[
            html.Div(
                className="editor-header",
                children=[
                    html.Div(filename),
                    html.Div([
                        html.Button("Diff", className="btn-control"),
                        html.Button("Original", className="btn-control"),
                        html.Button("Modified", className="btn-control", style={"color": "#fff"}),
                    ])
                ]
            ),
            html.Pre(
                content,
                className="editor-content",
                style={"whiteSpace": "pre-wrap"}
            )
        ]
    )

def create_browser_view(url: str = "about:blank") -> html.Iframe:
    """
    Create a browser view component.
    
    Args:
        url: URL to display in the browser frame
        
    Returns:
        Dash component for browser view
    """
    return html.Iframe(
        src=url,
        style={"width": "100%", "height": "100%", "border": "none"}
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

def create_view_type_indicator(view_type: str = "Editor", 
                              operation_type: str = "Working on", 
                              file_path: str = "") -> html.Div:
    """
    Create the view type indicator component.
    
    Args:
        view_type: Current view type (Editor, Terminal, Browser)
        operation_type: Current operation being performed
        file_path: Path of the file being operated on
        
    Returns:
        Dash component for view type indicator
    """
    return html.Div(
        id="view-type-indicator",
        style={
            "padding": "5px 10px",
            "backgroundColor": "#2d2d2d",
            "borderBottom": "1px solid #444",
            "display": "flex",
            "alignItems": "center"
        },
        children=[
            html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
            html.Span(id="current-view-type", children=view_type),
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
                    html.Span(id="operation-type", children=operation_type),
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
    )

def create_playback_controls() -> html.Div:
    """
    Create playback controls for the view container.
    
    Returns:
        Dash component for playback controls
    """
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
                        title="Previous step"
                    ),
                    html.Button(
                        html.I(id="play-icon", className="fas fa-play"),
                        className="btn-control",
                        id="playback-play",
                        title="Play/Pause"
                    ),
                    html.Button(
                        html.I(className="fas fa-step-forward"),
                        className="btn-control",
                        id="playback-forward",
                        title="Next step"
                    ),
                    html.Div(
                        id="playback-slider-container",
                        style={"width": "300px", "marginLeft": "10px", "marginRight": "10px"},
                        children=[
                            dcc.Slider(
                                id="playback-slider",
                                min=0,
                                max=100,
                                step=1,
                                value=0,
                                marks=None,
                                updatemode='drag'
                            )
                        ]
                    ),
                    html.Button(
                        html.I(className="fas fa-bolt"),
                        className="btn-control active",
                        id="live-button",
                        title="Live mode"
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
                    html.Span(id="task-progress", children="1/5", className="time-indicator")
                ]
            )
        ]
    )