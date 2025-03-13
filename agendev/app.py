"""
AgenDev - AI-driven Development Assistant

Entry point for the AgenDev application.
"""
import logging
from dash import Dash

# Import UI components
from UI.landing_page import create_landing_page
from UI.main_view import create_main_view
from UI.callbacks import register_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

# Set up the app layout with landing page and main view
app.layout = create_landing_page() 

# Register callbacks for app functionality
register_callbacks(app)

# Run the app
if __name__ == '__main__':
    logger.info("Starting AgenDev application")
    app.run_server(debug=True)
