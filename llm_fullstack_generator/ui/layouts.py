# ui/layouts.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    """Create the main application layout"""
    return dbc.Container([
        html.H1("LLM Fullstack Generator", className="mt-4 mb-4"),
        html.P("Generate a full-stack application using local LLM", className="lead"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Project Configuration"),
                    dbc.CardBody([
                        # Project name and description
                        dbc.FormGroup([
                            dbc.Label("Project Name", html_for="project-name"),
                            dbc.Input(id="project-name", type="text", placeholder="Enter project name")
                        ], className="mb-3"),
                        
                        dbc.FormGroup([
                            dbc.Label("Project Description", html_for="project-description"),
                            dbc.Textarea(
                                id="project-description",
                                placeholder="Describe your project in detail...",
                                style={"height": "150px"}
                            )
                        ], className="mb-3"),
                        
                        # Technology stack
                        dbc.FormGroup([
                            dbc.Label("Language", html_for="language-dropdown"),
                            dcc.Dropdown(
                                id="language-dropdown",
                                options=[
                                    {"label": "Python", "value": "python"},
                                    {"label": "NodeJS", "value": "nodejs"}
                                ],
                                value="python"
                            )
                        ], className="mb-3"),
                        
                        dbc.FormGroup([
                            dbc.Label("Framework (optional)", html_for="framework-input"),
                            dbc.Input(id="framework-input", type="text", placeholder="e.g., Flask, Express")
                        ], className="mb-3"),
                        
                        dbc.FormGroup([
                            dbc.Label("Database (optional)", html_for="database-dropdown"),
                            dcc.Dropdown(
                                id="database-dropdown",
                                options=[
                                    {"label": "None", "value": ""},
                                    {"label": "SQLite", "value": "sqlite"},
                                    {"label": "PostgreSQL", "value": "postgresql"},
                                    {"label": "MongoDB", "value": "mongodb"}
                                ],
                                value=""
                            )
                        ], className="mb-3"),
                        
                        # Frontend options
                        dbc.FormGroup([
                            dbc.Label("Include Frontend"),
                            dbc.Checklist(
                                id="frontend-toggle",
                                options=[{"label": "Yes", "value": True}],
                                value=[],
                                switch=True
                            )
                        ], className="mb-3"),
                        
                        dbc.FormGroup([
                            dbc.Label("Frontend Framework (if applicable)", html_for="frontend-framework"),
                            dbc.Input(
                                id="frontend-framework",
                                type="text",
                                placeholder="e.g., React, Vue",
                                disabled=True
                            )
                        ], className="mb-3"),
                        
                        # Generate button
                        dbc.Button("Generate Project", id="generate-button", color="primary", className="mt-2")
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Project Output"),
                    dbc.CardBody(
                        html.Div(id="project-output", className="p-2")
                    )
                ], className="h-100")
            ], width=6)
        ]),
        
        # Footer
        html.Footer([
            html.Hr(),
            html.P("LLM Fullstack Generator - Powered by Local LLM", className="text-center text-muted")
        ], className="mt-4")
    ], fluid=True)