# ui/callbacks.py
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

def register_callbacks(app, orchestrator):
    """Register all UI callbacks"""
    
    @app.callback(
        Output("frontend-framework", "disabled"),
        Input("frontend-toggle", "value")
    )
    def toggle_frontend_framework(frontend_toggle):
        """Enable/disable frontend framework input based on toggle"""
        return not frontend_toggle