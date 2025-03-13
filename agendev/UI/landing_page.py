"""
Landing Page Component for AgenDev.

This module contains the layout and logic for the AgenDev landing page.
"""
import logging
from dash import html, dcc
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

def create_landing_page():
    """
    Create the landing page layout.
    
    Returns:
        A Dash HTML Div containing the landing page layout.
    """
    logger.info("Creating landing page layout")
    
    return html.Div(
        id="app-container",
        children=[
            # Hidden div to store the current view
            dcc.Store(id="current-view", data="landing"),
            
            # Landing Page Content
            html.Div(
                id="landing-page",
                className="container",
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "height": "100vh",
                    "backgroundColor": "#1e1e1e",
                    "color": "#ccc",
                },
                children=[
                    # Logo and Slogan
                    html.Div(
                        className="text-center mb-4",
                        children=[
                            html.H1(
                                "AgenDev",
                                style={"color": "#61dafb", "fontSize": "3rem"}
                            ),
                            html.H3(
                                "Your AI Coding Partner",
                                style={"color": "#ccc", "marginBottom": "2rem"}
                            ),
                        ],
                    ),
                    
                    # Project Description Input
                    dbc.Card(
                        style={"width": "600px", "backgroundColor": "#2d2d2d", "border": "none"},
                        children=[
                            dbc.CardBody([
                                html.H5("What would you like to build today?", className="card-title"),
                                dbc.Textarea(
                                    id="project-description",
                                    placeholder="Describe your project (e.g., 'Build a Python web scraper for news sites')",
                                    style={"width": "100%", "height": "150px", "backgroundColor": "#3d3d3d", "color": "white"},
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Submit",
                                    id="submit-button",
                                    color="primary",
                                    className="mr-1",
                                    style={"backgroundColor": "#61dafb", "borderColor": "#61dafb"},
                                ),
                            ])
                        ],
                    ),
                ],
            ),
            
            # Main View (Hidden initially)
            html.Div(id="main-view-container", style={"display": "none"}),
        ],
    )
