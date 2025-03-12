"""AgenDev Dashboard - Main Application Module.

This is the main application module for the AgenDev Dashboard. It initializes the
AgenDev system and creates the Dash web application.
"""

import dash
import dash_bootstrap_components as dbc
import threading
import time
import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import AgenDev core components
from agendev.core import AgenDev, AgenDevConfig

# Import dashboard components
from dashboard.layout import create_dashboard_layout
from dashboard.callbacks import register_callbacks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the AgenDev system
try:
    config = AgenDevConfig(
        project_name="AgenDev Dashboard",
        llm_base_url="http://192.168.2.12:1234",
        tts_base_url="http://127.0.0.1:7860",
        notifications_enabled=True
    )
    agendev = AgenDev(config)
except Exception as e:
    logger.error(f"Error initializing AgenDev: {e}")
    # Create a minimal instance for UI rendering
    config = AgenDevConfig(project_name="AgenDev Dashboard")
    agendev = AgenDev(config)

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
app.title = "AgenDev Dashboard"
server = app.server

# Create the layout
app.layout = create_dashboard_layout(agendev)

# Register callbacks
register_callbacks(app, agendev)

# Enable notifications after startup
def enable_notifications_after_startup():
    time.sleep(1)  # Wait a moment for server to start
    if agendev and agendev.notification_manager:
        agendev.notification_manager.silence(False)
        agendev.notification_manager.info("AgenDev Dashboard is now running.")

# Run the application
if __name__ == "__main__":
    # Disable notifications during startup to prevent blocking
    if agendev and agendev.notification_manager:
        agendev.notification_manager.silence(True)
    
    # Start the server
    print("\n\n=== Starting AgenDev Dashboard ===")
    print(f"Project: {agendev.config.project_name}")
    
    # Print server info first so it's visible
    host = "127.0.0.1"
    port = 8050
    print(f"Server starting at: http://{host}:{port}/")
    print("Press Ctrl+C to quit")
    print("===============================\n\n")
    
    # Start notification enabling in a separate thread
    threading.Thread(target=enable_notifications_after_startup, daemon=True).start()
    
    # Start the server
    app.run_server(debug=False, host=host, port=port)