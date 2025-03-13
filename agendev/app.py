#!/usr/bin/env python3
"""
AgenDev - An AI-driven development assistant.

This is the main entry point for the AgenDev application, setting up the Dash app,
registering callbacks, and starting the server.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# Ensure necessary directories exist
Path("logs").mkdir(exist_ok=True)
Path("workspace").mkdir(exist_ok=True)
Path("artifacts/snapshots").mkdir(parents=True, exist_ok=True)
Path("artifacts/audio").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/agendev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("agendev")

# Import AgenDev components
try:
    from agendev.UI import (
        create_landing_page,
        create_main_view,
        create_stores,
        register_callbacks,
        init_containers
    )
    from agendev.llm_integration import LLMIntegration, LLMConfig
    from agendev.tts_notification import NotificationManager, NotificationConfig
    AGENDEV_AVAILABLE = True
    logger.info("AgenDev components imported successfully")
except ImportError as e:
    logger.error(f"Error importing AgenDev components: {e}")
    AGENDEV_AVAILABLE = False

def create_app():
    """
    Create and configure the Dash application.
    
    Returns:
        Configured Dash application
    """
    # Initialize Dash app with dark theme
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,  # Dark theme
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"  # Font Awesome
        ],
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    app.title = "AgenDev - AI-Driven Development Assistant"
    
    # Create stores for application state
    stores = create_stores()
    
    # Global stylesheet
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Global styles */
                :root {
                    --background-color: #1e1e1e;
                    --text-color: #ccc;
                    --accent-blue: #61dafb;
                    --accent-green: #00ff00;
                    --accent-purple: #800080;
                    --hover-color: #333;
                }
                
                body {
                    background-color: var(--background-color);
                    color: var(--text-color);
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                }
                
                /* Landing page styles */
                .landing-page {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .brand-logo {
                    font-size: 3.5rem;
                    margin-bottom: 10px;
                    color: var(--accent-blue);
                    font-weight: bold;
                }
                
                .brand-slogan {
                    font-size: 1.2rem;
                    margin-bottom: 40px;
                    color: var(--text-color);
                }
                
                .input-prompt {
                    width: 100%;
                    max-width: 600px;
                    padding: 20px;
                    border: 1px solid #444;
                    border-radius: 8px;
                    background-color: #2a2a2a;
                }
                
                .input-heading {
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: var(--accent-blue);
                }
                
                /* Main container styles */
                .main-container {
                    display: flex;
                    height: 100vh;
                    overflow: hidden;
                }
                
                .chat-container {
                    width: 40%;
                    padding: 20px;
                    overflow-y: auto;
                    border-right: 1px solid #444;
                }
                
                .view-container {
                    width: 60%;
                    display: flex;
                    flex-direction: column;
                }
                
                .view-header {
                    padding: 15px;
                    font-size: 1.2rem;
                    font-weight: bold;
                    background-color: #2a2a2a;
                    border-bottom: 1px solid #444;
                }
                
                .view-type-indicator {
                    padding: 10px 15px;
                    background-color: #333;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .view-content {
                    flex: 1;
                    padding: 15px;
                    overflow-y: auto;
                    background-color: #252525;
                }
                
                /* Playback controls */
                .playback-controls {
                    display: flex;
                    padding: 10px 15px;
                    background-color: #333;
                    border-top: 1px solid #444;
                    align-items: center;
                }
                
                .btn-control {
                    background-color: #444;
                    color: #fff;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 12px;
                    margin-right: 10px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .btn-control:hover {
                    background-color: var(--accent-blue);
                    color: #000;
                }
                
                .btn-control.active {
                    background-color: var(--accent-green);
                    color: #000;
                }
                
                .playback-slider-container {
                    flex: 1;
                    margin: 0 15px;
                    position: relative;
                }
                
                /* Task list styles */
                .collapsible-header {
                    padding: 10px;
                    border: 1px solid #444;
                    border-radius: 4px;
                    margin-bottom: 10px;
                    background-color: #2a2a2a;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                }
                
                .collapsible-header:hover {
                    background-color: var(--hover-color);
                }
                
                .collapsible-content {
                    padding: 10px;
                    border: 1px solid #444;
                    border-top: none;
                    border-radius: 0 0 4px 4px;
                    margin-top: -10px;
                    margin-bottom: 20px;
                    background-color: #252525;
                }
                
                /* Timeline markers */
                .timeline-marker {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                }
            </style>
            {%scripts%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Set up the application layout
    app.layout = html.Div([
        # Keyboard listener for playback navigation (using main-keyboard-listener to avoid conflicts)
        html.Div(id="main-keyboard-listener", style={"display": "none"}),
        
        # Data stores
        *stores,
        
        # Additional stores for keyboard events
        dcc.Store(id="keyboard-events", data={}),
        dcc.Store(id="keyboard-action", data=None),
        
        # Landing page
        html.Div(id="landing-page-container"),
        
        # Main view (initially hidden)
        html.Div(id="main-view-container", style={"display": "none"})
    ])
    
    # Register callbacks
    register_callbacks(app)
    
    # Initialize container contents
    init_containers(app)
    
    return app

def main():
    """Main entry point for the application."""
    if not AGENDEV_AVAILABLE:
        logger.error("AgenDev components not available. Exiting.")
        sys.exit(1)
    
    # Check LLM service availability
    try:
        llm = LLMIntegration(
            base_url="http://192.168.2.12:1234",
            config=LLMConfig(model="qwen2.5-7b-instruct")
        )
        result = llm.check_availability()
        if result.get("available", False):
            logger.info(f"LLM service available at http://192.168.2.12:1234")
        else:
            logger.warning(f"LLM service not available: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.warning(f"Could not check LLM service availability: {e}")
    
    # Check TTS service availability
    try:
        notification_config = NotificationConfig(
            enabled=True,
            history_enabled=True,
            history_file="artifacts/audio/notification_history.json"
        )
        notification_manager = NotificationManager(
            tts_base_url="http://127.0.0.1:7860",
            config=notification_config
        )
        status = notification_manager.get_connection_status()
        if status.get("tts_available", False):
            logger.info(f"TTS service available at http://127.0.0.1:7860")
        else:
            logger.warning(f"TTS service not available: {status.get('error', 'Unknown error')}")
    except Exception as e:
        logger.warning(f"Could not check TTS service availability: {e}")
    
    # Create application
    app = create_app()
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get("AGENDEV_HOST", "0.0.0.0")
    port = int(os.environ.get("AGENDEV_PORT", "8050"))
    
    # Run the server
    logger.info(f"Starting AgenDev server on {host}:{port}")
    app.run_server(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()