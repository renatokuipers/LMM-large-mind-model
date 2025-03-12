"""Dashboard layout definition for AgenDev."""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from .styles import styles
from .components import (
    create_chat_interface, create_card, create_project_form,
    create_overview_section, create_tasks_section, create_implementation_section,
    create_advanced_features_section
)

def create_dashboard_layout(agendev):
    """Create the main dashboard layout."""
    return html.Div([
        # Header
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row([
                            dbc.Col(html.I(className="fas fa-robot fa-lg")),
                            dbc.Col(dbc.NavbarBrand("AgenDev Dashboard", className="ms-2")),
                        ], align="center"),
                        href="/",
                        style={"textDecoration": "none"}
                    ),
                    html.Div([
                        dbc.Badge(
                            "Autonomous AI Development",
                            color="primary",
                            className="me-2"
                        ),
                        dbc.Badge(
                            id="autonomous-status-badge",
                            color="secondary",
                            children="Inactive",
                            className="me-3"
                        ),
                        # Settings button
                        html.Div([
                            dbc.Button(
                                html.I(className="fas fa-cog"),
                                color="link",
                                id="settings-button",
                                size="sm",
                                className="text-light p-0"
                            ),
                        ], style={"cursor": "pointer"})
                    ], className="d-flex align-items-center")
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
            style={
                "background": "linear-gradient(90deg, #1a1a2e 0%, #16213e 100%)",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.2)"
            }
        ),
        
        # Main content
        dbc.Container([
            dbc.Row([
                # Left column - Chat interface
                dbc.Col([
                    create_chat_interface()
                ], md=5, style={"height": "calc(100vh - 120px)"}),
                
                # Right column - Dashboard panels
                dbc.Col([
                    # Project has not started - show creation form
                    html.Div(
                        create_card("Create New Project", create_project_form(), color="accent_purple"),
                        id="project-form-container"
                    ),
                    
                    # Project info (only shown when project exists)
                    html.Div([
                        # Top row - Overview
                        dbc.Row([
                            dbc.Col([
                                create_overview_section(agendev)
                            ], md=12)
                        ]),
                        
                        # Bottom row - Tasks and Implementation
                        dbc.Row([
                            dbc.Col([
                                create_tasks_section(agendev)
                            ], md=6),
                            dbc.Col([
                                create_implementation_section()
                            ], md=6)
                        ]),
                        
                        # Advanced Features Row
                        dbc.Row([
                            dbc.Col([
                                create_advanced_features_section(agendev)
                            ], md=12, className="mt-4")
                        ])
                    ], id="project-info-container")
                ], md=7, style={"height": "calc(100vh - 120px)", "overflowY": "auto"})
            ])
        ], fluid=True),
        
        # Hidden elements for state management
        dcc.Store(id="active-page", data="overview"),
        dcc.Store(id="chat-history", data={"messages": []}),
        dcc.Store(id="app-settings", data={}),  # Store for app settings
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 seconds
            n_intervals=0
        ),

        # Settings Modal
        dbc.Modal(
            [
                dbc.ModalHeader(
                    html.H4("Settings", className="mb-0"),
                    close_button=True,
                    style={"background": "linear-gradient(90deg, #1a1a2e 0%, #16213e 100%)"}
                ),
                dbc.ModalBody([
                    # General settings section
                    html.Div([
                        html.H5("General Settings", className="border-bottom border-secondary pb-2 mb-3"),
                        
                        # Project Name
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Project Name"),
                                dbc.Input(
                                    id="settings-project-name",
                                    type="text",
                                    placeholder="Enter project name",
                                    value=agendev.config.project_name
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # Notification settings
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Notifications"),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Enable voice notifications", "value": 1}
                                    ],
                                    value=[1] if agendev.config.notifications_enabled else [],
                                    id="settings-notifications-enabled",
                                    switch=True,
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # Refresh interval
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("UI Refresh Interval (seconds)"),
                                dbc.Input(
                                    id="settings-refresh-interval",
                                    type="number",
                                    min=1,
                                    max=60,
                                    step=1,
                                    value=5
                                ),
                            ])
                        ], className="mb-3"),
                    ], className="mb-4"),
                    
                    # Advanced settings section
                    html.Div([
                        html.H5("Advanced Settings", className="border-bottom border-secondary pb-2 mb-3"),
                        
                        # LLM API URL
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("LLM API URL"),
                                dbc.Input(
                                    id="settings-llm-url",
                                    type="text",
                                    placeholder="http://localhost:1234",
                                    value=agendev.config.llm_base_url
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # TTS API URL
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("TTS API URL"),
                                dbc.Input(
                                    id="settings-tts-url",
                                    type="text",
                                    placeholder="http://localhost:7860",
                                    value=agendev.config.tts_base_url
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # Default Model
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Default LLM Model"),
                                dbc.Input(
                                    id="settings-default-model",
                                    type="text",
                                    placeholder="Model name",
                                    value=agendev.config.default_model
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # Auto-save settings
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Auto-save"),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Enable auto-save", "value": 1}
                                    ],
                                    value=[1] if agendev.config.auto_save else [],
                                    id="settings-auto-save",
                                    switch=True,
                                ),
                            ])
                        ], className="mb-2"),
                        
                        # Auto-save interval (only visible if auto-save enabled)
                        html.Div([
                            dbc.Label("Auto-save Interval (minutes)"),
                            dbc.Input(
                                id="settings-auto-save-interval",
                                type="number",
                                min=1,
                                max=60,
                                step=1,
                                value=agendev.config.auto_save_interval_minutes
                            ),
                        ], id="auto-save-interval-container", className="mb-3"),
                    ]),
                    
                    # Appearance settings section
                    html.Div([
                        html.H5("Appearance", className="border-bottom border-secondary pb-2 mb-3"),
                        
                        # Theme selector
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Color Theme"),
                                dbc.Select(
                                    id="settings-color-theme",
                                    options=[
                                        {"label": "Blue & Purple (Default)", "value": "blue-purple"},
                                        {"label": "Green & Teal", "value": "green-teal"},
                                        {"label": "Red & Orange", "value": "red-orange"}
                                    ],
                                    value="blue-purple"
                                ),
                            ])
                        ], className="mb-3"),
                        
                        # Font size
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Font Size"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Small", "value": "small"},
                                        {"label": "Medium", "value": "medium"},
                                        {"label": "Large", "value": "large"}
                                    ],
                                    value="medium",
                                    id="settings-font-size",
                                    inline=True
                                ),
                            ])
                        ], className="mb-3"),
                    ], className="mb-4"),
                    
                    # Alert for settings status
                    html.Div(id="settings-alert")
                ], style={"background": styles['background']}),
                dbc.ModalFooter([
                    dbc.Button("Reset to Defaults", color="secondary", id="settings-reset", className="me-auto"),
                    dbc.Button("Save", color="primary", id="settings-save")
                ], style={"background": "linear-gradient(90deg, #1a1a2e 0%, #16213e 100%)"})
            ],
            id="settings-modal",
            size="lg",
            scrollable=True,
            style={"color": styles['text']},
            contentClassName="border border-secondary"
        ),
    ], style={
        "background": styles['background'],
        "minHeight": "100vh",
        "color": styles['text']
    }) 