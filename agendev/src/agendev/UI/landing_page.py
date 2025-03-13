"""
Landing page module for the AgenDev UI.

This module contains the layout for the initial landing page with the project input form.
"""
from dash import html, dcc

def create_landing_page() -> html.Div:
    """
    Create the landing page layout.
    
    Returns:
        Dash component for landing page
    """
    return html.Div(
        id="landing-page",
        className="landing-page",
        children=[
            html.Div(className="brand-logo", children=["AgenDev"]),
            html.Div(
                className="brand-slogan", 
                children=["An Intelligent Agentic Development System"]
            ),
            html.Div(
                className="input-prompt",
                children=[
                    html.H2("What would you like to develop today?", className="input-heading"),
                    dcc.Textarea(
                        id="initial-prompt",
                        placeholder="Describe your project or what you'd like help with...",
                        style={
                            "width": "100%",
                            "height": "120px",
                            "borderRadius": "4px",
                            "padding": "10px",
                            "marginBottom": "15px",
                            "backgroundColor": "#2a2a2a",
                            "color": "#fff",
                            "border": "1px solid #444"
                        }
                    ),
                    html.Button(
                        "Submit",
                        id="submit-button",
                        style={
                            "width": "100%",
                            "padding": "10px",
                            "borderRadius": "4px",
                            "backgroundColor": "#61dafb",
                            "color": "#000",
                            "border": "none",
                            "cursor": "pointer",
                            "fontWeight": "bold"
                        }
                    )
                ]
            )
        ]
    )