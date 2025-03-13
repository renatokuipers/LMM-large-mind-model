import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime

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

# Custom CSS for styling to match the screenshots
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
                --primary-color: #333;
                --secondary-color: #444;
                --text-color: #f8f9fa;
                --accent-color: #61dafb;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --warning-color: #ffc107;
            }
            
            body {
                background-color: #1a1a1a;
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
                background-color: #1a1a1a;
                border-right: 1px solid #333;
            }
            
            .view-container {
                width: 50%;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                background-color: #1a1a1a;
            }
            
            .view-header {
                background-color: #333;
                padding: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .view-content {
                flex-grow: 1;
                overflow: auto;
                padding: 0;
                background-color: #2a2a2a;
            }
            
            .view-controls {
                background-color: #333;
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
                background-color: #333;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .user-message {
                padding: 15px;
                background-color: #2a2a2a;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .collapsible-header {
                display: flex;
                align-items: center;
                padding: 10px;
                background-color: #333;
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            
            .collapsible-header:hover {
                background-color: #444;
            }
            
            .collapsible-content {
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 4px;
                margin-bottom: 15px;
                margin-left: 15px;
                border-left: 2px solid #61dafb;
            }
            
            .command-element {
                background-color: #2a2a2a;
                padding: 8px 12px;
                border-radius: 4px;
                margin: 5px 0;
                font-family: 'Consolas', 'Courier New', monospace;
                border-left: 3px solid #61dafb;
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
                background-color: #61dafb;
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
                background-color: #28a745;
                color: #fff;
            }
            
            .status-tag.in-progress {
                background-color: #ffc107;
                color: #000;
            }
            
            .status-tag.error {
                background-color: #dc3545;
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
                color: #fff;
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
                background-color: #333;
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
                color: #61dafb;
            }
            
            .brand-slogan {
                font-size: 1rem;
                color: #888;
                margin-bottom: 30px;
                text-align: center;
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

# Landing page layout with centered input
landing_page = html.Div(
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

# Terminal view component
def create_terminal_view(content):
    return html.Div(
        className="terminal-view",
        children=[
            html.Pre(content)
        ]
    )

# Editor view component
def create_editor_view(filename, content, language="text"):
    syntax_highlighting = {
        "python": {
            "keywords": ["def", "class", "import", "from", "return", "if", "else", "elif", "for", "while", "try", "except", "with"],
            "keyword_color": "#569CD6",
            "string_color": "#CE9178",
            "comment_color": "#6A9955",
            "function_color": "#DCDCAA",
            "variable_color": "#9CDCFE"
        },
        "json": {
            "keywords": ["null", "true", "false"],
            "keyword_color": "#569CD6",
            "string_color": "#CE9178",
            "number_color": "#B5CEA8",
            "punctuation_color": "#D4D4D4"
        },
        "text": {
            "color": "#D4D4D4"
        }
    }
    
    return html.Div(
        className="editor-view",
        children=[
            html.Div(
                className="editor-header",
                children=[
                    html.Div(filename),
                    html.Div([
                        html.Button("Diff", className="btn-control"),
                        html.Button("Original", className="btn-control"),
                        html.Button("Modified", className="btn-control", style={"color": "#fff"}),
                    ])
                ]
            ),
            html.Pre(
                content,
                className="editor-content",
                style={"whiteSpace": "pre-wrap"}
            )
        ]
    )

# Collapsible section component
def create_collapsible_section(id_prefix, header_content, content, is_open=True):
    return html.Div([
        html.Div(
            className="collapsible-header",
            id=f"{id_prefix}-header",
            children=[
                html.I(
                    className="fas fa-chevron-down mr-2",
                    style={"marginRight": "10px"}
                ),
                header_content
            ]
        ),
        html.Div(
            id=f"{id_prefix}-content",
            className="collapsible-content",
            style={"display": "block" if is_open else "none"},
            children=content
        )
    ])

# Command execution component
def create_command_element(command, status="completed"):
    icon_class = "fas fa-check-circle text-success" if status == "completed" else "fas fa-spinner fa-spin text-warning"
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span("Executing command", style={"marginRight": "10px"}),
            html.Code(command, className="command-element")
        ]
    )

# File creation/editing component
def create_file_operation(operation, filepath, status="completed"):
    icon_class = "fas fa-check-circle text-success" if status == "completed" else "fas fa-spinner fa-spin text-warning"
    
    return html.Div(
        className="status-element",
        children=[
            html.Span(className=f"status-icon {icon_class}"),
            html.Span(f"{operation} file", style={"marginRight": "10px"}),
            html.Code(filepath, className="file-path")
        ]
    )

# Main split view layout
main_view = html.Div(
    id="main-container",
    className="main-container",
    style={"display": "none"},  # Hidden initially
    children=[
        # Left side - Chat interface
        html.Div(
            className="chat-container",
            id="chat-container",
            children=[
                # Header with current task
                html.Div(
                    className="system-message",
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "marginBottom": "15px"},
                            children=[
                                html.I(className="fas fa-robot", style={"fontSize": "24px", "marginRight": "10px", "color": "#61dafb"}),
                                html.Span("AgenDev", style={"fontSize": "24px", "fontWeight": "bold"})
                            ]
                        ),
                        html.H3(id="project-title", children="Python Snake Game Development", style={"margin": "0 0 10px 0"})
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task1",
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"marginRight": "10px", "color": "#28a745"}),
                        html.Span("Continue: Create Next.js application for political debate simulation")
                    ]),
                    [
                        html.P("Creating a Next.js application for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_command_element("cd /home/ubuntu && create_nextjs_app zelenskyy_debate_sim"),
                        html.P("Successfully created Next.js application for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/app/data/scenarios.json")
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task2",
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"marginRight": "10px", "color": "#28a745"}),
                        html.Span("Develop dialogue system and political scenario scripts")
                    ]),
                    [
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Editing", "zelenskyy_debate_sim/src/app/page.tsx"),
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/app/simulation/page.tsx"),
                        html.P("Implementing dialogue system and scenarios for the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Editing", "zelenskyy_debate_sim/src/app/globals.css")
                    ]
                ),
                
                # Example of a task section
                create_collapsible_section(
                    "task3",
                    html.Div([
                        html.I(className="fas fa-spinner fa-spin", style={"marginRight": "10px", "color": "#ffc107"}),
                        html.Span("Design and implement user interface with styling")
                    ]),
                    [
                        html.P("Moving to add additional user interface elements and styling to the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/CharacterPortrait.tsx"),
                        html.P("Adding user interface components and styling to enhance the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/DialogueBubble.tsx"),
                        html.P("Adding user interface components and styling to enhance the Zelenskyy-Trump-Vance interactive simulation."),
                        create_file_operation("Creating", "zelenskyy_debate_sim/src/components/ResponseOption.tsx")
                    ]
                ),
                
                # Thinking indicator
                html.Div(
                    className="chat-message",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "color": "#888"
                            },
                            children=[
                                html.I(className="fas fa-circle-notch fa-spin", style={"marginRight": "10px"}),
                                html.Span("Thinking")
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Right side - Dynamic view (Terminal, Editor, Browser)
        html.Div(
            className="view-container",
            children=[
                # Header
                html.Div(
                    className="view-header",
                    children=[
                        html.Div("AgenDev's Computer"),
                        html.Button(
                            html.I(className="fas fa-expand"),
                            className="btn-control"
                        )
                    ]
                ),
                
                # View type indicator
                html.Div(
                    style={
                        "padding": "5px 10px",
                        "backgroundColor": "#2d2d2d",
                        "borderBottom": "1px solid #444",
                        "display": "flex",
                        "alignItems": "center"
                    },
                    children=[
                        html.Span("AgenDev is using", style={"color": "#888", "marginRight": "5px"}),
                        html.Span("Editor"),
                        html.Div(
                            style={
                                "marginLeft": "20px",
                                "display": "flex",
                                "alignItems": "center",
                                "color": "#888",
                                "fontSize": "0.85em"
                            },
                            children=[
                                html.Span("Creating file"),
                                html.Code(
                                    "zelenskyy_debate_sim/src/app/data/scenarios.json",
                                    style={
                                        "marginLeft": "5px",
                                        "backgroundColor": "transparent",
                                        "padding": "0"
                                    }
                                )
                            ]
                        )
                    ]
                ),
                
                # Content area (can be terminal, editor, or browser)
                html.Div(
                    className="view-content",
                    id="view-content",
                    children=[
                        # Default to editor view
                        create_editor_view(
                            "scenarios.json",
                            '''
{
  "scenarios": [
    {
      "id": 1,
      "title": "Opening Remarks",
      "description": "President Trump welcomes you to the White House. The meeting has just begun with initial pleasantries.",
      "trumpDialogue": "President Trump welcomes you to the White House. We're going to have a great discussion today about ending this terrible war. I hope I'm going to be remembered as a peacemaker.",
      "vanceDialogue": "",
      "options": [
        {
          "id": "1a",
          "text": "Thank you, Mr. President. Ukraine is grateful for America's support. We look forward to discussing how we can achieve a just peace that ensures Ukraine's security.",
          "type": "diplomatic",
          "trumpReaction": "positive",
          "vanceReaction": "neutral",
          "nextScenario": 2
        },
        {
          "id": "1b",
          "text": "Thank you for meeting with me. I must emphasize that Ukraine needs more than just words - we need continued military support and security guarantees to end this war.",
          "type": "assertive",
          "trumpReaction": "neutral",
          "vanceReaction": "negative",
          "nextScenario": 2
        },
        ...
      ]
    }
  ]
}''',
                            "json"
                        )
                    ]
                ),
                
                # Controls
                html.Div(
                    className="view-controls",
                    children=[
                        html.Div(
                            className="progress-controls",
                            children=[
                                html.Button(
                                    html.I(className="fas fa-step-backward"),
                                    className="btn-control",
                                    id="playback-backward"
                                ),
                                html.Button(
                                    html.I(className="fas fa-play"),
                                    className="btn-control",
                                    id="playback-play"
                                ),
                                html.Button(
                                    html.I(className="fas fa-step-forward"),
                                    className="btn-control",
                                    id="playback-forward"
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="playback-slider",
                                        min=0,
                                        max=100,
                                        value=50,
                                        updatemode="drag",
                                        marks=None,
                                        tooltip={"always_visible": False},
                                        className="timeline-slider"
                                    ),
                                    style={"width": "300px", "marginLeft": "10px", "marginRight": "10px"}
                                )
                            ]
                        ),
                        html.Div(
                            className="status-indicator",
                            children=[
                                html.Span(
                                    html.I(className="fas fa-check-circle"),
                                    className="status-tag success",
                                    style={"marginRight": "5px"}
                                ),
                                html.Span("Deploy simulation to a public URL for permanent access"),
                                html.Span("9/9", className="time-indicator")
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Store of playback data with timeline steps
def create_demo_playback_data():
    return {
        "total_steps": 10,
        "current_step": 0,
        "is_playing": False,
        "steps": [
            {
                "type": "terminal",
                "content": "ubuntu@sandbox:~ $ cd /home/ubuntu && cd /home/ubuntu\nubuntu@sandbox:~ $ create_nextjs_app python_snake_game\nStarting setup...\nCreating Next.js app for development: python_snake_game\nInstalling dependencies...\nInitializing git repository...",
                "timestamp": "00:00"
            },
            {
                "type": "terminal",
                "content": "ubuntu@sandbox:~ $ cd /home/ubuntu && cd /home/ubuntu\nubuntu@sandbox:~ $ create_nextjs_app python_snake_game\nStarting setup...\nCreating Next.js app for development: python_snake_game\nInstalling dependencies...\nInitializing git repository...\nCreated new Next.js app python_snake_game at /home/ubuntu/python_snake_game\n--- Project Structure ---\n|— migrations/\n|   └— 0001_initial.sql      # DB migration script\n|— src/\n|   |— app/                 # Next.js pages\n|   |   └— counter.ts       # Example component\n|   |— components/\n|   |— hooks/\n|   |— lib/\n|   └— wrangler.toml        # Cloudflare config",
                "timestamp": "00:10"
            },
            {
                "type": "editor",
                "filename": "game_engine.py",
                "content": "# game_engine.py\n\nclass SnakeGame:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n        self.snake = [(width // 2, height // 2)]\n        self.direction = 'RIGHT'\n        self.food = None\n        self.score = 0\n        self.game_over = False\n        self._place_food()\n    \n    def _place_food(self):\n        # Logic to place food\n        import random\n        while True:\n            x = random.randint(0, self.width - 1)\n            y = random.randint(0, self.height - 1)\n            if (x, y) not in self.snake:\n                self.food = (x, y)\n                break",
                "timestamp": "00:30"
            },
            {
                "type": "editor",
                "filename": "snake_game.py",
                "content": "# snake_game.py\nimport pygame\nimport sys\nfrom game_engine import SnakeGame\n\nclass SnakeGameUI:\n    def __init__(self, width=20, height=20, cell_size=20):\n        self.width = width\n        self.height = height\n        self.cell_size = cell_size\n        self.game = SnakeGame(width, height)\n        \n        # Initialize pygame\n        pygame.init()\n        self.screen = pygame.display.set_mode(\n            (width * cell_size, height * cell_size)\n        )\n        pygame.display.set_caption('Python Snake Game')\n        \n        # Colors\n        self.colors = {\n            'background': (15, 15, 15),\n            'snake': (0, 255, 0),\n            'food': (255, 0, 0),\n            'text': (255, 255, 255)\n        }\n        \n        # Game clock\n        self.clock = pygame.time.Clock()\n        self.speed = 10  # FPS",
                "timestamp": "00:45"
            },
            {
                "type": "editor",
                "filename": "snake_game.py",
                "content": "# snake_game.py\nimport pygame\nimport sys\nfrom game_engine import SnakeGame\n\nclass SnakeGameUI:\n    def __init__(self, width=20, height=20, cell_size=20):\n        self.width = width\n        self.height = height\n        self.cell_size = cell_size\n        self.game = SnakeGame(width, height)\n        \n        # Initialize pygame\n        pygame.init()\n        self.screen = pygame.display.set_mode(\n            (width * cell_size, height * cell_size)\n        )\n        pygame.display.set_caption('Python Snake Game')\n        \n        # Colors\n        self.colors = {\n            'background': (15, 15, 15),\n            'snake': (0, 255, 0),\n            'food': (255, 0, 0),\n            'text': (255, 255, 255)\n        }\n        \n        # Game clock\n        self.clock = pygame.time.Clock()\n        self.speed = 10  # FPS\n        \n    def draw(self):\n        # Clear screen\n        self.screen.fill(self.colors['background'])\n        \n        # Draw snake\n        for segment in self.game.snake:\n            pygame.draw.rect(\n                self.screen,\n                self.colors['snake'],\n                pygame.Rect(\n                    segment[0] * self.cell_size,\n                    segment[1] * self.cell_size,\n                    self.cell_size,\n                    self.cell_size\n                )\n            )\n        \n        # Draw food\n        pygame.draw.rect(\n            self.screen,\n            self.colors['food'],\n            pygame.Rect(\n                self.game.food[0] * self.cell_size,\n                self.game.food[1] * self.cell_size,\n                self.cell_size,\n                self.cell_size\n            )\n        )\n        \n        # Update display\n        pygame.display.flip()",
                "timestamp": "01:05"
            },
            {
                "type": "editor",
                "filename": "main.py",
                "content": "# main.py\nfrom snake_game import SnakeGameUI\nimport pygame\nimport sys\n\ndef main():\n    # Create game instance\n    game_ui = SnakeGameUI(width=20, height=20, cell_size=30)\n    \n    # Main game loop\n    while not game_ui.game.game_over:\n        # Process events\n        for event in pygame.event.get():\n            if event.type == pygame.QUIT:\n                pygame.quit()\n                sys.exit()\n            elif event.type == pygame.KEYDOWN:\n                if event.key == pygame.K_UP and game_ui.game.direction != 'DOWN':\n                    game_ui.game.direction = 'UP'\n                elif event.key == pygame.K_DOWN and game_ui.game.direction != 'UP':\n                    game_ui.game.direction = 'DOWN'\n                elif event.key == pygame.K_LEFT and game_ui.game.direction != 'RIGHT':\n                    game_ui.game.direction = 'LEFT'\n                elif event.key == pygame.K_RIGHT and game_ui.game.direction != 'LEFT':\n                    game_ui.game.direction = 'RIGHT'\n        \n        # Update game state\n        game_ui.game.update()\n        \n        # Draw game\n        game_ui.draw()\n        \n        # Control game speed\n        game_ui.clock.tick(game_ui.speed)\n    \n    # Game over screen\n    game_ui.show_game_over()\n    \n    # Wait for quit event\n    waiting = True\n    while waiting:\n        for event in pygame.event.get():\n            if event.type == pygame.QUIT:\n                waiting = False\n    \n    pygame.quit()\n\nif __name__ == \"__main__\":\n    main()",
                "timestamp": "01:30"
            },
            {
                "type": "terminal",
                "content": "ubuntu@sandbox:~ $ cd /home/ubuntu/python_snake_game\nubuntu@sandbox:~/python_snake_game $ python main.py\nTraceback (most recent call last):\n  File \"main.py\", line 31, in <module>\n    main()\n  File \"main.py\", line 19, in main\n    game_ui.game.update()\nAttributeError: 'SnakeGame' object has no attribute 'update'\n",
                "timestamp": "02:00"
            },
            {
                "type": "editor",
                "filename": "game_engine.py",
                "content": "# game_engine.py\n\nclass SnakeGame:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n        self.snake = [(width // 2, height // 2)]\n        self.direction = 'RIGHT'\n        self.food = None\n        self.score = 0\n        self.game_over = False\n        self._place_food()\n    \n    def _place_food(self):\n        # Logic to place food\n        import random\n        while True:\n            x = random.randint(0, self.width - 1)\n            y = random.randint(0, self.height - 1)\n            if (x, y) not in self.snake:\n                self.food = (x, y)\n                break\n                \n    def update(self):\n        # Move snake based on current direction\n        head_x, head_y = self.snake[0]\n        \n        if self.direction == 'UP':\n            head_y -= 1\n        elif self.direction == 'DOWN':\n            head_y += 1\n        elif self.direction == 'LEFT':\n            head_x -= 1\n        elif self.direction == 'RIGHT':\n            head_x += 1\n            \n        # Check for game over conditions\n        if (head_x < 0 or head_x >= self.width or\n            head_y < 0 or head_y >= self.height or\n            (head_x, head_y) in self.snake):\n            self.game_over = True\n            return\n            \n        # Check if snake ate food\n        if (head_x, head_y) == self.food:\n            self.score += 1\n            self._place_food()\n        else:\n            # Remove tail if snake didn't eat\n            self.snake.pop()\n            \n        # Add new head\n        self.snake.insert(0, (head_x, head_y))",
                "timestamp": "02:30"
            },
            {
                "type": "terminal",
                "content": "ubuntu@sandbox:~ $ cd /home/ubuntu/python_snake_game\nubuntu@sandbox:~/python_snake_game $ python main.py\n[Game is now running successfully in a pygame window]",
                "timestamp": "03:00"
            },
            {
                "type": "editor",
                "filename": "README.md",
                "content": "# Python Snake Game\n\nA classic snake game implemented in Python using Pygame.\n\n## Features\n\n- Clean, modular code structure with separation of game logic and UI\n- Smooth controls using arrow keys\n- Score tracking\n- Game over detection\n\n## Requirements\n\n- Python 3.6+\n- Pygame\n\n## Installation\n\n```bash\npip install pygame\n```\n\n## How to Run\n\n```bash\npython main.py\n```\n\n## Controls\n\n- Arrow keys to change direction\n- Esc to quit\n\n## Project Structure\n\n- `main.py` - Entry point for the game\n- `game_engine.py` - Core game logic\n- `snake_game.py` - UI and rendering logic\n\n## Future Improvements\n\n- Add pause functionality\n- Add high score tracking\n- Implement difficulty levels\n- Add sound effects",
                "timestamp": "03:15"
            }
        ]
    }

# Add a Store component to manage playback state
playback_store = dcc.Store(
    id='playback-data',
    data=create_demo_playback_data()
)

# Add app state store
app_state_store = dcc.Store(
    id='app-state',
    data={"view": "landing", "initial_prompt": ""}
)

# Set the app layout - this is what was missing
app.layout = html.Div([
    app_state_store,
    playback_store,
    landing_page,
    main_view
])

# Callback to transition from landing page to main view
@app.callback(
    [Output("app-state", "data"),
     Output("landing-page", "style"),
     Output("main-container", "style"),
     Output("project-title", "children")],
    [Input("submit-button", "n_clicks")],
    [State("initial-prompt", "value"),
     State("app-state", "data")],
    prevent_initial_call=True
)
def transition_to_main_view(n_clicks, prompt_value, current_state):
    if not n_clicks:
        raise PreventUpdate
    
    # Update state
    current_state["view"] = "main"
    current_state["initial_prompt"] = prompt_value
    
    # Generate a title based on the prompt
    title = "New Project"
    if prompt_value:
        # Simple algorithm to extract a title
        if "create" in prompt_value.lower() and "snake" in prompt_value.lower() and "python" in prompt_value.lower():
            title = "Python Snake Game Development"
        elif "todo" in prompt_value.lower() or "task" in prompt_value.lower() or "list" in prompt_value.lower():
            title = "Todo List Application"
        elif "dashboard" in prompt_value.lower() or "data" in prompt_value.lower() or "visualization" in prompt_value.lower():
            title = "Data Visualization Dashboard"
        elif "web" in prompt_value.lower() or "site" in prompt_value.lower() or "app" in prompt_value.lower():
            title = "Web Application Development"
        elif "game" in prompt_value.lower():
            title = "Game Development Project"
        elif "api" in prompt_value.lower() or "backend" in prompt_value.lower() or "server" in prompt_value.lower():
            title = "API Development Project"
        else:
            # Extract key words for a generic title
            words = prompt_value.split()
            if len(words) > 3:
                # Take a few significant words from the middle of the prompt
                middle_index = len(words) // 2
                title_words = words[max(0, middle_index-1):min(len(words), middle_index+2)]
                title = " ".join(word.capitalize() for word in title_words) + " Project"
            else:
                # For short prompts, use the whole thing
                title = prompt_value.capitalize()
    
    # Hide landing page, show main container
    landing_style = {"display": "none"}
    main_style = {"display": "flex"}
    
    return current_state, landing_style, main_style, title

# Callback for task1 collapsible section
@app.callback(
    Output("task1-content", "style"),
    Input("task1-header", "n_clicks"),
    State("task1-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task1(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Callback for task2 collapsible section  
@app.callback(
    Output("task2-content", "style"),
    Input("task2-header", "n_clicks"),
    State("task2-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task2(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Callback for task3 collapsible section
@app.callback(
    Output("task3-content", "style"),
    Input("task3-header", "n_clicks"),
    State("task3-content", "style"),
    prevent_initial_call=True
)
def toggle_section_task3(n_clicks, current_style):
    if not n_clicks:
        raise PreventUpdate
    
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}
    return new_style

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)