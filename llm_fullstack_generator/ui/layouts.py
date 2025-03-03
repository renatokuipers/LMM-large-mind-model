# ui/layouts.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    """Create the main application layout"""
    return dbc.Container([
        html.H1("LLM Fullstack Generator", className="mt-4 mb-4 text-primary"),
        html.P("Generate a full-stack application using local LLM", className="lead text-light"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Project Configuration", className="mb-0")),
                    dbc.CardBody([
                        # Project name and description
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Project Name", html_for="project-name", className="text-light"),
                                dbc.Input(
                                    id="project-name", 
                                    type="text", 
                                    placeholder="Enter project name",
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Project Description", html_for="project-description", className="text-light"),
                                dbc.Textarea(
                                    id="project-description",
                                    placeholder="Describe your project in detail...",
                                    style={"height": "150px"},
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                        ]),
                        
                        # Technology stack
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Language", html_for="language-dropdown", className="text-light"),
                                dcc.Dropdown(
                                    id="language-dropdown",
                                    options=[
                                        {"label": "Python", "value": "python"},
                                        {"label": "NodeJS", "value": "nodejs"}
                                    ],
                                    value="python",
                                    className="bg-dark text-light",
                                    style={
                                        "color": "white",
                                        "background-color": "#303030"
                                    }
                                )
                            ], className="mb-3"),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Framework (optional)", html_for="framework-input", className="text-light"),
                                dbc.Input(
                                    id="framework-input", 
                                    type="text", 
                                    placeholder="e.g., Flask, Express",
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Database (optional)", html_for="database-dropdown", className="text-light"),
                                dcc.Dropdown(
                                    id="database-dropdown",
                                    options=[
                                        {"label": "None", "value": ""},
                                        {"label": "SQLite", "value": "sqlite"},
                                        {"label": "PostgreSQL", "value": "postgresql"},
                                        {"label": "MongoDB", "value": "mongodb"}
                                    ],
                                    value="",
                                    className="bg-dark text-light",
                                    style={
                                        "color": "white",
                                        "background-color": "#303030"
                                    }
                                )
                            ], className="mb-3"),
                        ]),
                        
                        # Frontend options
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Include Frontend", className="text-light"),
                                dbc.Checklist(
                                    id="frontend-toggle",
                                    options=[{"label": "Yes", "value": True}],
                                    value=[],
                                    switch=True
                                )
                            ], className="mb-3"),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Frontend Framework (if applicable)", html_for="frontend-framework", className="text-light"),
                                dbc.Input(
                                    id="frontend-framework",
                                    type="text",
                                    placeholder="e.g., React, Vue",
                                    disabled=True,
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                        ]),
                        
                        # Generate button
                        dbc.Button("Generate Project", id="generate-button", color="primary", className="mt-2")
                    ])
                ], className="bg-dark border-secondary")
            ], width=5),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Project Output", className="mb-0")),
                    dbc.CardBody([
                        # Add a progress bar
                        dbc.Progress(
                            id="generation-progress", 
                            value=0,
                            striped=True,
                            animated=True,
                            className="mb-3"
                        ),
                        
                        # Project output
                        html.Div(id="project-output", className="p-2"),
                        
                        # Log output
                        html.H6("Generation Log:", className="mt-4 mb-2"),
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(id="generation-log", className="text-light", 
                                        style={"maxHeight": "300px", "overflowY": "auto"})
                            ),
                            className="bg-dark border-secondary"
                        ),
                        
                        # Interval for status updates
                        dcc.Interval(
                            id="status-interval",
                            interval=1000,  # 1 second
                            n_intervals=0
                        )
                    ])
                ], className="h-100 bg-dark border-secondary")
            ], width=7)
        ]),
        
        # Footer
        html.Footer([
            html.Hr(className="border-secondary mt-4"),
            html.P("LLM Fullstack Generator - Powered by Local LLM", className="text-center text-muted")
        ])
    ], fluid=True, className="bg-dark text-light")