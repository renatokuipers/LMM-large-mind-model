"""
AgenDev - An Intelligent Agentic Development System.

This is the main entry point for the AgenDev application.
"""
import dash
import dash_bootstrap_components as dbc

# Import UI components
from src.agendev.UI import create_landing_page, create_main_view, create_stores, register_callbacks

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

# Setup custom CSS for the dark theme - this would ideally be in a separate assets/custom.css file
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AgenDev - Intelligent Agentic Development System</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #1e1e1e;
                --secondary-color: #2a2a2a;
                --tertiary-color: #333;
                --text-color: #fff;
                --text-secondary: #ccc;
                --text-muted: #888;
                --accent-color: #61dafb;
                --success-color: #00ff00;
                --warning-color: #ffc107;
                --danger-color: #dc3545;
                --purple-accent: #800080;
            }
            
            body {
                background-color: var(--primary-color);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                overflow: hidden;
                margin: 0;
                padding: 0;
                height: 100vh;
            }
            
            .landing-page {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
            }
            
            .main-container {
                display: flex;
                height: 100vh;
                width: 100%;
                overflow: hidden;
            }
            
            .chat-container {
                width: 50%;
                height: 100%;
                overflow-y: auto;
                padding: 20px;
                background-color: var(--primary-color);
                border-right: 1px solid var(--tertiary-color);
            }
            
            .view-container {
                width: 50%;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                background-color: var(--primary-color);
            }
            
            .view-header {
                background-color: var(--tertiary-color);
                padding: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .view-content {
                flex-grow: 1;
                overflow: auto;
                padding: 0;
                background-color: var(--secondary-color);
            }
            
            .view-controls {
                background-color: var(--tertiary-color);
                padding: 10px;
                display: flex;
                justify-content: space-between;
            }
            
            .chat-message {
                margin-bottom: 20px;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .system-message {
                padding: 15px;
                background-color: var(--tertiary-color);
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .user-message {
                padding: 15px;
                background-color: var(--secondary-color);
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .collapsible-header {
                display: flex;
                align-items: center;
                padding: 10px;
                background-color: var(--tertiary-color);
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            
            .collapsible-header:hover {
                background-color: #444;
            }
            
            .collapsible-content {
                padding: 10px;
                background-color: var(--secondary-color);
                border-radius: 4px;
                margin-bottom: 15px;
                margin-left: 15px;
                border-left: 2px solid var(--accent-color);
            }
            
            .command-element {
                background-color: var(--secondary-color);
                padding: 8px 12px;
                border-radius: 4px;
                margin: 5px 0;
                font-family: 'Consolas', 'Courier New', monospace;
                border-left: 3px solid var(--accent-color);
            }
            
            .status-element {
                display: flex;
                align-items: center;
                margin: 5px 0;
            }
            
            .status-icon {
                margin-right: 10px;
            }
            
            .terminal-view {
                background-color: #1e1e1e;
                color: #ddd;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 10px;
                height: 100%;
                overflow: auto;
                white-space: pre-wrap;
            }
            
            .editor-view {
                background-color: #1e1e1e;
                height: 100%;
                overflow: auto;
            }
            
            .editor-header {
                background-color: #2d2d2d;
                padding: 5px 10px;
                border-bottom: 1px solid #444;
                display: flex;
                justify-content: space-between;
            }
            
            .editor-content {
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                color: #ddd;
                min-height: calc(100% - 40px);
            }
            
            .browser-view {
                background-color: #fff;
                height: 100%;
                overflow: auto;
            }
            
            .file-path {
                font-family: 'Consolas', 'Courier New', monospace;
                color: #888;
                font-size: 0.85em;
                margin-bottom: 5px;
            }
            
            .function-tag {
                background-color: var(--accent-color);
                color: #000;
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 5px;
                font-size: 0.8em;
            }
            
            .status-tag {
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 5px;
                font-size: 0.8em;
            }
            
            .status-tag.success {
                background-color: var(--success-color);
                color: #000;
            }
            
            .status-tag.in-progress {
                background-color: var(--warning-color);
                color: #000;
            }
            
            .status-tag.error {
                background-color: var(--danger-color);
                color: #fff;
            }
            
            .progress-controls {
                display: flex;
                align-items: center;
            }
            
            .time-indicator {
                font-size: 0.8em;
                color: #888;
                margin-left: 10px;
            }
            
            .btn-control {
                background: none;
                border: none;
                color: #888;
                font-size: 1em;
                cursor: pointer;
                padding: 5px;
                transition: color 0.2s;
            }
            
            .btn-control:hover {
                color: var(--accent-color);
            }
            
            .btn-control.active {
                color: var(--success-color);
            }
            
            .code-content {
                border-radius: 4px;
                background-color: #2d2d2d;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                overflow-x: auto;
            }
            
            .input-prompt {
                width: 80%;
                max-width: 800px;
                padding: 20px;
                background-color: var(--tertiary-color);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .input-heading {
                font-size: 1.5rem;
                margin-bottom: 20px;
                text-align: center;
            }
            
            .brand-logo {
                margin-bottom: 30px;
                font-size: 2.5rem;
                font-weight: bold;
                color: var(--accent-color);
            }
            
            .brand-slogan {
                font-size: 1rem;
                color: var(--text-muted);
                margin-bottom: 30px;
                text-align: center;
            }
            
            .todo-markdown h1 {
                color: var(--accent-color);
                font-size: 1.8rem;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            
            .todo-markdown h2 {
                color: var(--accent-color);
                font-size: 1.4rem;
                margin-top: 0.8rem;
                margin-bottom: 0.4rem;
            }
            
            .todo-markdown h3 {
                color: var(--accent-color);
                font-size: 1.2rem;
                margin-top: 0.6rem;
                margin-bottom: 0.3rem;
            }
            
            .todo-markdown ul {
                padding-left: 20px;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            
            .todo-markdown li {
                margin-bottom: 0.3rem;
            }
            
            .todo-markdown input[type="checkbox"] {
                margin-right: 0.5rem;
            }
            
            .todo-markdown input[type="checkbox"]:checked + span {
                text-decoration: line-through;
                opacity: 0.7;
            }
        </style>
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

# Combine all components into the app layout
app.layout = dash.html.Div([
    # Stores for application state
    *create_stores(),
    
    # Landing page
    create_landing_page(),
    
    # Main view (initially hidden)
    create_main_view()
])

# Register callbacks
register_callbacks(app)

# Set title
app.title = "AgenDev - Intelligent Agentic Development System"

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)