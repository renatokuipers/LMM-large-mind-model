# chat.py
import os
import time
import threading
import base64
from typing import List, Dict, Optional, Any, Callable

# Dash components
import dash
from dash import html, dcc, callback, Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, MultiplexerTransform, BlockingCallbackTransform
import diskcache

# Custom modules
from llm_module import LLMClient, Message
from tts_module import (
    text_to_speech, 
    get_available_voices, 
    play_audio,
    get_output_path,
    GenerateAudioRequest,
    TTSClient
)

# Additional imports for enhanced functionality
import json
import datetime
from pathlib import Path
import markdown
import traceback
from dash.exceptions import PreventUpdate

# Initialize the disk cache for long callbacks
cache = diskcache.Cache("./cache")

# Initialize LLM client
llm_client = LLMClient()

# Theme and styling
# Define custom dark theme with yellow and blue accents
DARK_THEME = {
    'dark': True,
    'primary': '#FFD700',  # Golden yellow
    'secondary': '#4169E1',  # Royal blue
    'background': '#121212',  # Very dark gray
    'surface': '#1E1E1E',  # Dark gray
    'text': '#FFFFFF',  # White text
    'accent': '#29B6F6',  # Light blue accent
}

# Define CSS for custom styling and gradients
custom_css = '''
/* Dark mode background with gradient */
body {
    background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
    color: #FFFFFF;
    min-height: 100vh;
    margin: 0;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* Chat container */
.chat-container {
    background-color: rgba(30, 30, 30, 0.7);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    padding: 20px;
    margin: 20px 0;
    border: 1px solid rgba(255, 215, 0, 0.1);
}

/* Message styling */
.user-message {
    background: linear-gradient(90deg, rgba(65, 105, 225, 0.1) 0%, rgba(65, 105, 225, 0.3) 100%);
    border-left: 4px solid #4169E1;
    border-radius: 0 15px 15px 0;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
    margin-left: auto;
}

.assistant-message {
    background: linear-gradient(90deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 215, 0, 0.3) 100%);
    border-left: 4px solid #FFD700;
    border-radius: 0 15px 15px 0;
    padding: 10px 15px;
    margin: 10px 0;
    max-width: 85%;
}

/* Input box styling */
.message-input {
    background-color: rgba(30, 30, 30, 0.7);
    border: 1px solid rgba(255, 215, 0, 0.3);
    border-radius: 10px;
    color: white;
    padding: 12px;
    transition: all 0.3s ease;
}

.message-input:focus {
    border-color: #FFD700;
    box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2);
    outline: none;
}

/* Button styling */
.send-button {
    background: linear-gradient(90deg, #4169E1 0%, #29B6F6 100%);
    border: none;
    border-radius: 10px;
    color: white;
    padding: 12px 20px;
    transition: all 0.3s ease;
}

.send-button:hover {
    background: linear-gradient(90deg, #5a7dfa 0%, #47c4fa 100%);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(41, 182, 246, 0.4);
}

/* Settings panel */
.settings-panel {
    background-color: rgba(30, 30, 30, 0.8);
    border-radius: 15px;
    padding: 15px;
    margin-top: 20px;
    border: 1px solid rgba(255, 215, 0, 0.1);
}

/* Slider styling */
.rc-slider-track {
    background-color: #FFD700;
}

.rc-slider-handle {
    border-color: #FFD700;
    background-color: #FFD700;
}

.rc-slider-rail {
    background-color: #333333;
}

/* Tabs styling */
.dash-tab {
    background-color: #1E1E1E;
    color: #BBBBBB;
    border-color: #333;
    border-radius: 5px 5px 0 0;
    padding: 10px 15px;
}

.dash-tab--selected {
    background: linear-gradient(90deg, #4169E1 0%, #29B6F6 100%);
    color: white;
    border: none;
    font-weight: 500;
}

/* Dropdown styling */
.dash-dropdown .Select-control {
    background-color: #1E1E1E;
    border-color: #333;
    color: white;
}

.dash-dropdown .Select-menu-outer {
    background-color: #1E1E1E;
    border-color: #333;
    color: white;
}

.dash-dropdown .Select-value-label {
    color: white !important;
}

/* Voice block styling */
.voice-block {
    background: linear-gradient(90deg, rgba(65, 105, 225, 0.1) 0%, rgba(65, 105, 225, 0.3) 100%);
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.3s ease;
}

.voice-block:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(65, 105, 225, 0.2);
}

/* Audio player styling */
.audio-player {
    width: 100%;
    margin: 10px 0;
    background-color: rgba(30, 30, 30, 0.5);
    border-radius: 10px;
    padding: 5px;
}

/* Loading animation */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 215, 0, 0.3);
    border-radius: 50%;
    border-top-color: #FFD700;
    animation: spin 1s ease-in-out infinite;
    margin-left: 10px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Tooltip styling */
.tooltip {
    background-color: #1E1E1E;
    color: white;
    border: 1px solid #FFD700;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 0.85rem;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1E1E1E; 
}
 
::-webkit-scrollbar-thumb {
    background: #4169E1; 
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #29B6F6; 
}

/* Markdown styling */
.markdown-content {
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

.markdown-content h1, .markdown-content h2, .markdown-content h3, 
.markdown-content h4, .markdown-content h5, .markdown-content h6 {
    color: #FFD700;
    margin-top: 20px;
    margin-bottom: 10px;
}

.markdown-content p {
    margin-bottom: 15px;
}

.markdown-content a {
    color: #29B6F6;
    text-decoration: none;
}

.markdown-content a:hover {
    text-decoration: underline;
}

.markdown-content pre {
    background-color: #2d2d2d;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    margin: 15px 0;
    border-left: 4px solid #4169E1;
}

.markdown-content code {
    font-family: 'Consolas', 'Monaco', monospace;
    background-color: #2d2d2d;
    padding: 3px 5px;
    border-radius: 3px;
    font-size: 0.9em;
}

.markdown-content ul, .markdown-content ol {
    margin-left: 25px;
    margin-bottom: 15px;
}

.markdown-content blockquote {
    border-left: 4px solid #4169E1;
    padding-left: 15px;
    margin-left: 0;
    color: #cccccc;
}

.markdown-content table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}

.markdown-content th, .markdown-content td {
    border: 1px solid #444;
    padding: 8px 12px;
    text-align: left;
}

.markdown-content th {
    background-color: #2d2d2d;
    color: #FFD700;
}

.markdown-content tr:nth-child(even) {
    background-color: #2d2d2d;
}
'''

# Initialize the Dash app with the proper callback transforms
app = DashProxy(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY, 
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    transforms=[
        MultiplexerTransform(),
        BlockingCallbackTransform()
    ]
)

# Set the app title
app.title = "AI Chat with TTS"

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        ''' + custom_css + '''
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

# Initialize the conversation history
conversation_history = []

# Global variable for current message
current_message = ""

# Define available models
available_models = [
    "qwen2.5-7b-instruct",
    "qwen2.5-72b-instruct",
    "llama3-8b-instruct",
    "llama3-70b-instruct",
    "claude-3-sonnet-20240229",
    "gpt-3.5-turbo"
]

# Store current settings
app_settings = {
    "model": "qwen2.5-7b-instruct",
    "temperature": 0.7,
    "max_tokens": -1,
    "voice_enabled": True,
    "auto_play_voice": True,
    "voice": "af_bella",
    "speech_speed": 1.0,
    "base_url": "http://192.168.2.12:1234",
    "system_prompt": "You are a helpful, creative, and friendly AI assistant. Answer concisely unless asked to elaborate."
}

# Layout Components

# Message component to display chat messages
def create_message_component(message, role):
    """Create a message component with optional TTS controls and markdown support"""
    message_class = "user-message" if role == "user" else "assistant-message"
    
    # Render markdown for assistant messages
    message_content = None
    if role == "assistant" and message:
        try:
            # Convert markdown to HTML
            html_message = markdown.markdown(
                message,
                extensions=['fenced_code', 'tables', 'nl2br']
            )
            message_content = html.Div(
                dangerously_allow_html=True,
                children=html_message,
                className="markdown-content"
            )
        except Exception:
            # Fallback to plain text if markdown parsing fails
            message_content = html.Div(message)
    else:
        # Display user messages as plain text
        message_content = html.Div(message)
    
    # For assistant messages, add TTS button
    tts_controls = None
    if role == "assistant" and app_settings["voice_enabled"] and message.strip():
        tts_controls = html.Div([
            html.Button(
                [html.I(className="fas fa-volume-up")], 
                id={"type": "play-tts", "index": len(conversation_history)},
                className="btn btn-sm btn-outline-primary ml-2",
                title="Play text to speech"
            ),
            html.Audio(
                id={"type": "audio-player", "index": len(conversation_history)},
                className="audio-player d-none",
                controls=True
            )
        ], className="d-flex align-items-center mt-2")
    
    return html.Div([
        html.Div([
            html.Strong(role.capitalize() + ": ", className="mr-2"),
            html.Div(message_content, id={"type": "message-content", "index": len(conversation_history)})
        ]),
        tts_controls
    ], className=message_class)

# Chat history component
chat_history = html.Div(
    id="chat-history",
    children=[],
    style={
        "height": "calc(100vh - 300px)",
        "overflowY": "auto",
        "padding": "10px",
    },
    className="chat-container"
)

# Input area with send button
chat_input_area = dbc.Form(
    [
        dbc.Row([
            dbc.Col([
                dbc.Input(
                    id="message-input",
                    type="text",
                    placeholder="Type your message here...",
                    className="message-input",
                    autoFocus=True,
                    n_submit=0
                ),
            ], width=10),
            dbc.Col([
                dbc.Button(
                    [
                        "Send ",
                        html.I(className="fas fa-paper-plane ml-1")
                    ],
                    id="send-button",
                    color="primary",
                    className="send-button w-100"
                ),
            ], width=2),
        ]),
    ],
    className="mt-3"
)

# LLM settings panel
llm_settings = dbc.Card([
    dbc.CardHeader("LLM Settings"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Model"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[{"label": model, "value": model} for model in available_models],
                    value=app_settings["model"],
                    clearable=False,
                    className="dash-dropdown"
                )
            ], width=6),
            dbc.Col([
                html.Label("API Base URL"),
                dbc.Input(
                    id="base-url-input",
                    type="text",
                    value=app_settings["base_url"],
                    className="message-input"
                )
            ], width=6)
        ]),
        html.Div([
            html.Label("Temperature: " + str(app_settings["temperature"])),
            dcc.Slider(
                id="temperature-slider",
                min=0.0,
                max=2.0,
                step=0.1,
                marks={i/10: str(i/10) for i in range(0, 21, 5)},
                value=app_settings["temperature"]
            )
        ], className="mt-3"),
        html.Div([
            html.Label("Max Tokens: " + str(app_settings["max_tokens"])),
            dcc.Slider(
                id="max-tokens-slider",
                min=100,
                max=8000,
                step=100,
                marks={i: str(i) for i in range(0, 8001, 2000)},
                value=app_settings["max_tokens"]
            )
        ], className="mt-3"),
        html.Div([
            html.Label("System Prompt"),
            dbc.Textarea(
                id="system-prompt-input",
                value=app_settings["system_prompt"],
                className="message-input",
                style={"height": "100px"}
            )
        ], className="mt-3")
    ]),
], className="settings-panel")

# TTS settings panel
tts_settings = dbc.Card([
    dbc.CardHeader("Text-to-Speech Settings"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Voice Options:"),
                dcc.Checklist(
                    id="voice-enabled-checkbox",
                    options=[{"label": "Enable Text-to-Speech", "value": "enabled"}],
                    value=["enabled"] if app_settings["voice_enabled"] else [],
                    className="ml-2"
                ),
            ], width=6),
            dbc.Col([
                html.Label("Playback Options:"),
                dcc.Checklist(
                    id="auto-play-checkbox",
                    options=[{"label": "Auto-play TTS", "value": "auto_play"}],
                    value=["auto_play"] if app_settings["auto_play_voice"] else [],
                    className="ml-2"
                ),
            ], width=6)
        ]),
        html.Div([
            html.Label("Voice"),
            dcc.Dropdown(
                id="voice-dropdown",
                options=[{"label": voice, "value": voice} for voice in get_available_voices()],
                value=app_settings["voice"],
                clearable=False,
                className="dash-dropdown"
            )
        ], className="mt-3"),
        html.Div([
            html.Label("Speech Speed: " + str(app_settings["speech_speed"])),
            dcc.Slider(
                id="speech-speed-slider",
                min=0.1,
                max=2.0,
                step=0.1,
                marks={i/10: str(i/10) for i in range(1, 21, 5)},
                value=app_settings["speech_speed"]
            )
        ], className="mt-3"),
        html.Div([
            html.Button(
                "Test Voice",
                id="test-voice-button",
                className="send-button mt-3"
            ),
            html.Div(id="test-voice-output")
        ])
    ]),
], className="settings-panel mt-3")

# Settings panel with tabs
settings_panel = html.Div([
    dbc.Collapse(
        dbc.Tabs([
            dbc.Tab(llm_settings, label="LLM Settings", tab_id="llm-tab", className="dash-tab"),
            dbc.Tab(tts_settings, label="TTS Settings", tab_id="tts-tab", className="dash-tab"),
        ], active_tab="llm-tab"),
        id="settings-collapse",
        is_open=False,
    ),
    dbc.Button(
        [
            html.I(className="fas fa-cog mr-2"),
            "Settings"
        ],
        id="settings-toggle",
        color="secondary",
        className="mt-3"
    ),
], className="mt-3")

# Streaming response indicator
streaming_indicator = html.Div([
    html.Div([
        "AI is thinking",
        html.Div(className="loading-spinner ml-2")
    ], className="d-flex align-items-center")
], id="streaming-indicator", style={"display": "none"})

# Prompt suggestions
prompt_suggestions = html.Div(
    id="prompt-suggestions",
    className="mt-2"
)

# Layout
app.layout = dbc.Container([
    html.H1([
        "AI Chat",
        html.Span(" with TTS", style={"color": DARK_THEME["primary"]}),
    ], className="mt-4 mb-4 text-center"),
    
    # Notification area for errors
    html.Div(id="notification-area", className="mt-2"),
    
    # Main chat interface
    dbc.Row([
        dbc.Col([
            # Chat history
            chat_history,
            
            # Streaming indicator
            streaming_indicator,
            
            # Prompt suggestions
            prompt_suggestions,
            
            # Input area
            chat_input_area,
            
            # Additional buttons row
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-cog mr-2"),
                            "Settings"
                        ],
                        id="settings-toggle",
                        color="secondary",
                        className="mt-3 w-100"
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-file-export mr-2"),
                            "Export Chat"
                        ],
                        id="export-chat-button",
                        color="primary",
                        className="mt-3 w-100"
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-file-import mr-2"),
                            "Import Chat"
                        ],
                        id="import-chat-button",
                        color="primary",
                        className="mt-3 w-100"
                    ),
                    dcc.Upload(
                        id="upload-chat-json",
                        children=html.Div([]),
                        style={"display": "none"},
                        multiple=False
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-search mr-2"),
                            "Search Chat"
                        ],
                        id="search-chat-button",
                        color="info",
                        className="mt-3 w-100"
                    ),
                ], width=3),
            ]),
            
            # Search modal
            dbc.Modal([
                dbc.ModalHeader("Search Conversation"),
                dbc.ModalBody([
                    dbc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Enter search term...",
                        className="message-input mb-3"
                    ),
                    html.Div(id="search-results")
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-search-modal", className="ml-auto")
                ),
            ], id="search-modal", is_open=False),
            
            # Settings
            dbc.Collapse(
                dbc.Tabs([
                    dbc.Tab(llm_settings, label="LLM Settings", tab_id="llm-tab", className="dash-tab"),
                    dbc.Tab(tts_settings, label="TTS Settings", tab_id="tts-tab", className="dash-tab"),
                ], active_tab="llm-tab"),
                id="settings-collapse",
                is_open=False,
            ),
            
            # Store for conversation history
            dcc.Store(id="conversation-store"),
            
            # Interval for stream updates
            dcc.Interval(id='stream-update-interval', interval=50, disabled=True),
            
            # Hidden audio elements for TTS
            html.Div(id="hidden-audio-output", style={"display": "none"}),
            
            # Hidden div for download trigger
            html.Div(id="download-trigger", style={"display": "none"}),
            dcc.Download(id="download-chat"),
            
            # Keyboard shortcut help tooltip
            html.Div([
                dbc.Button(
                    html.I(className="fas fa-keyboard"),
                    id="keyboard-shortcuts-btn",
                    size="sm",
                    color="link",
                    className="position-fixed bottom-0 end-0 mb-2 mr-2"
                ),
                dbc.Tooltip([
                    html.H5("Keyboard Shortcuts"),
                    html.Hr(),
                    html.P([html.Kbd("Ctrl"), " + ", html.Kbd("Enter"), ": Send message"]),
                    html.P([html.Kbd("Ctrl"), " + ", html.Kbd(","), ": Toggle settings"]),
                    html.P([html.Kbd("Ctrl"), " + ", html.Kbd("S"), ": Export chat"]),
                    html.P([html.Kbd("Ctrl"), " + ", html.Kbd("O"), ": Import chat"]),
                    html.P([html.Kbd("Escape"), ": Close open panels"]),
                ], 
                target="keyboard-shortcuts-btn",
                placement="top"
                ),
            ]),
            
        ], width={"size": 10, "offset": 1}),
    ]),
    
    # Footer
    html.Footer([
        html.Hr(),
        html.P([
            "Powered by Qwen and TTS API | ",
            html.A("Reset Chat", id="reset-chat-button", href="#"),
            " | ",
            f"Version 1.0.0 ({datetime.datetime.now().strftime('%Y-%m-%d')})"
        ])
    ], className="text-center mt-5"),
    
], fluid=True)

# Auto-scroll function in JavaScript
app.clientside_callback(
    """
    function scrollToBottom(n_children) {
        var chatHistory = document.getElementById('chat-history');
        if (chatHistory) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-history", "children", allow_duplicate=True),
    Input("chat-history", "children"),
    prevent_initial_call=True
)

# Update settings based on UI inputs
@callback(
    [
        Output("temperature-slider", "value"),
        Output("max-tokens-slider", "value"),
        Output("model-dropdown", "value"),
        Output("base-url-input", "value"),
        Output("system-prompt-input", "value"),
        Output("voice-dropdown", "value"),
        Output("speech-speed-slider", "value"),
        Output("voice-enabled-checkbox", "value"),
        Output("auto-play-checkbox", "value")
    ],
    [
        Input("temperature-slider", "value"),
        Input("max-tokens-slider", "value"),
        Input("model-dropdown", "value"),
        Input("base-url-input", "value"),
        Input("system-prompt-input", "value"),
        Input("voice-dropdown", "value"),
        Input("speech-speed-slider", "value"),
        Input("voice-enabled-checkbox", "value"),
        Input("auto-play-checkbox", "value")
    ],
    [State("conversation-store", "data")]
)
def update_settings(
    temperature, max_tokens, model, base_url, system_prompt,
    voice, speech_speed, voice_enabled, auto_play, conversation_data
):
    # Update app settings
    app_settings["temperature"] = temperature
    app_settings["max_tokens"] = max_tokens
    app_settings["model"] = model
    app_settings["base_url"] = base_url
    app_settings["system_prompt"] = system_prompt
    app_settings["voice"] = voice
    app_settings["speech_speed"] = speech_speed
    app_settings["voice_enabled"] = "enabled" in voice_enabled if voice_enabled else False
    app_settings["auto_play_voice"] = "auto_play" in auto_play if auto_play else False
    
    # Update LLM client base URL if changed
    if base_url != llm_client.base_url:
        llm_client.base_url = base_url
    
    # Return the updated values
    return (
        temperature, 
        max_tokens, 
        model, 
        base_url, 
        system_prompt, 
        voice, 
        speech_speed, 
        ["enabled"] if app_settings["voice_enabled"] else [],
        ["auto_play"] if app_settings["auto_play_voice"] else []
    )

# Toggle settings panel
@callback(
    Output("settings-collapse", "is_open"),
    [Input("settings-toggle", "n_clicks")],
    [State("settings-collapse", "is_open")],
)
def toggle_settings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Test voice button callback
@callback(
    Output("test-voice-output", "children"),
    Input("test-voice-button", "n_clicks"),
    prevent_initial_call=True
)
def test_voice(n_clicks):
    if n_clicks:
        try:
            # Generate a test audio file
            test_text = "This is a test of the text-to-speech system with the selected voice and speed settings."
            result = text_to_speech(
                text=test_text,
                voice=app_settings["voice"],
                speed=app_settings["speech_speed"],
                auto_play=True
            )
            
            return html.Div([
                html.P("Voice test successful!", className="text-success mt-2"),
                html.Audio(
                    src=f"data:audio/wav;base64,{get_base64_audio(result['audio_path'])}",
                    controls=True,
                    className="audio-player mt-2"
                )
            ])
        except Exception as e:
            return html.P(f"Error testing voice: {str(e)}", className="text-danger mt-2")
    
    return None

def get_base64_audio(file_path):
    """Convert audio file to base64 for embedding in page"""
    with open(file_path, "rb") as audio_file:
        encoded = base64.b64encode(audio_file.read())
        return encoded.decode()

# Reset chat
@callback(
    [
        Output("chat-history", "children", allow_duplicate=True),
        Output("conversation-store", "data", allow_duplicate=True)
    ],
    Input("reset-chat-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_chat(n_clicks):
    # Clear conversation history
    global conversation_history
    conversation_history = []
    
    return [], []

# Send message on button click or Enter key
@callback(
    [
        Output("message-input", "value"),
        Output("message-input", "n_submit"),
        Output("send-button", "disabled"),
        Output("stream-update-interval", "disabled"),
        Output("streaming-indicator", "style")
    ],
    [
        Input("send-button", "n_clicks"),
        Input("message-input", "n_submit")
    ],
    [
        State("message-input", "value"),
        State("stream-update-interval", "disabled")
    ],
    prevent_initial_call=True
)
def send_message_start(n_clicks, n_submit, message, stream_disabled):
    if not message or message.strip() == "":
        return "", 0, False, True, {"display": "none"}
    
    # Store the message in a global variable to access it in the streaming callback
    # This is critical for the streaming process to work correctly
    global current_message
    current_message = message
    
    # Start the streaming process by enabling the interval
    return "", 0, True, False, {"display": "block"}

# Streaming process using dash_extensions blocking callback
@callback(
    [
        Output("chat-history", "children", allow_duplicate=True),
        Output("conversation-store", "data", allow_duplicate=True),
        Output("send-button", "disabled", allow_duplicate=True),
        Output("stream-update-interval", "disabled", allow_duplicate=True),
        Output("streaming-indicator", "style", allow_duplicate=True),
        Output("hidden-audio-output", "children", allow_duplicate=True),
        Output("notification-area", "children", allow_duplicate=True),
    ],
    [
        Input("stream-update-interval", "disabled"),
    ],
    [
        State("chat-history", "children"),
        State("conversation-store", "data")
    ],
    prevent_initial_call=True,
)
def process_message_with_error_handling(stream_disabled, history, conversation_data):
    # Add visual indicators that would have been handled by 'running' parameter
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, False, True, {"display": "none"}, dash.no_update, dash.no_update
    
    if stream_disabled:
        # Interval is disabled, nothing to do
        return dash.no_update, dash.no_update, False, True, {"display": "none"}, dash.no_update, dash.no_update
    
    # Use the global current_message
    global current_message
    message = current_message
    
    # Initialize conversation data if None
    if conversation_data is None:
        conversation_data = []
    
    # Add user message to history
    user_message_component = create_message_component(message, "user")
    history = history + [user_message_component] if history else [user_message_component]
    
    # Add user message to conversation data
    user_message = {"role": "user", "content": message}
    conversation_data.append(user_message)
    
    # Create messages list for the LLM
    messages = [
        Message(role="system", content=app_settings["system_prompt"])
    ]
    
    # Add conversation history
    for msg in conversation_data:
        messages.append(Message(role=msg["role"], content=msg["content"]))
    
    notification = None
    audio_element = None
    
    # Get streamed response
    try:
        # Initialize the streaming response
        response = llm_client.chat_completion(
            messages=messages,
            model=app_settings["model"],
            temperature=app_settings["temperature"],
            max_tokens=app_settings["max_tokens"],
            stream=True
        )
        
        # Create a placeholder for the assistant's message
        assistant_response = ""
        assistant_message_component = create_message_component(assistant_response, "assistant")
        history = history + [assistant_message_component]
        
        # Process the stream in chunks
        for chunk in llm_client.stream_generator(response):
            assistant_response += chunk
            # Update the last message (assistant's response)
            history[-1] = create_message_component(assistant_response, "assistant")
            time.sleep(0.01)  # Small delay to avoid overwhelming the UI
            
        # Add assistant response to conversation data
        assistant_message = {"role": "assistant", "content": assistant_response}
        conversation_data.append(assistant_message)
        
        # Handle TTS if enabled
        if app_settings["voice_enabled"] and assistant_response.strip():
            try:
                # Generate TTS for the assistant's response
                tts_result = text_to_speech(
                    text=assistant_response,
                    voice=app_settings["voice"],
                    speed=app_settings["speech_speed"],
                    auto_play=app_settings["auto_play_voice"]
                )
                
                # Create audio element
                if "audio_path" in tts_result and os.path.exists(tts_result["audio_path"]):
                    audio_element = html.Audio(
                        src=f"data:audio/wav;base64,{get_base64_audio(tts_result['audio_path'])}",
                        id="current-audio",
                        autoPlay=app_settings["auto_play_voice"],
                        controls=True,
                        style={"display": "none"}
                    )
            except Exception as e:
                tts_error = f"TTS Error: {str(e)}"
                print(tts_error)
                notification = html.Div(tts_error, className="alert alert-warning")
        
    except Exception as e:
        # Handle error in the response
        error_details = traceback.format_exc()
        print(f"Error in LLM response: {str(e)}\n{error_details}")
        
        error_message = f"Error communicating with the language model: {str(e)}"
        history = history + [create_message_component(error_message, "assistant")]
        
        # Add error message to conversation data
        error_system_message = {"role": "system", "content": f"Error occurred: {str(e)}"}
        conversation_data.append(error_system_message)
        
        notification = html.Div([
            html.Strong("Error: "), 
            error_message
        ], className="alert alert-danger")
    
    # Return updated history and conversation
    return history, conversation_data, False, True, {"display": "none"}, audio_element, notification

# Handle playing TTS for specific messages
@callback(
    Output({"type": "audio-player", "index": dash.MATCH}, "src"),
    Output({"type": "audio-player", "index": dash.MATCH}, "className"),
    Input({"type": "play-tts", "index": dash.MATCH}, "n_clicks"),
    State({"type": "message-content", "index": dash.MATCH}, "children"),
    prevent_initial_call=True
)
def play_message_tts(n_clicks, message_content):
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    try:
        # Generate TTS for the message
        tts_result = text_to_speech(
            text=message_content,
            voice=app_settings["voice"],
            speed=app_settings["speech_speed"],
            auto_play=False  # Don't auto-play, we'll handle it with the audio element
        )
        
        if "audio_path" in tts_result and os.path.exists(tts_result["audio_path"]):
            audio_src = f"data:audio/wav;base64,{get_base64_audio(tts_result['audio_path'])}"
            return audio_src, "audio-player"
    
    except Exception as e:
        print(f"TTS Error: {str(e)}")
    
    return dash.no_update, dash.no_update

# Add export chat functionality
@callback(
    Output("download-chat", "data"),
    Input("export-chat-button", "n_clicks"),
    State("conversation-store", "data"),
    prevent_initial_call=True
)
def export_chat(n_clicks, conversation_data):
    if not n_clicks or not conversation_data:
        raise PreventUpdate
    
    try:
        # Create chat export with metadata
        export_data = {
            "metadata": {
                "exported_at": datetime.datetime.now().isoformat(),
                "version": "1.0.0",
                "model": app_settings["model"]
            },
            "settings": app_settings,
            "conversation": conversation_data
        }
        
        # Format the current date and time for filename
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_export_{now}.json"
        
        return dict(
            content=json.dumps(export_data, indent=2),
            filename=filename,
            type="application/json"
        )
    except Exception as e:
        print(f"Export error: {str(e)}")
        return dash.no_update

# Add import chat functionality
@callback(
    [
        Output("chat-history", "children", allow_duplicate=True),
        Output("conversation-store", "data", allow_duplicate=True),
        Output("notification-area", "children"),
    ],
    Input("upload-chat-json", "contents"),
    State("upload-chat-json", "filename"),
    prevent_initial_call=True
)
def import_chat(contents, filename):
    if contents is None:
        raise PreventUpdate
    
    try:
        # Decode the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        data = json.loads(decoded)
        
        # Validate the format
        if "conversation" not in data:
            return dash.no_update, dash.no_update, html.Div(
                "Invalid chat file format",
                className="alert alert-danger"
            )
        
        # Create new message components for each message
        history = []
        conversation_data = data["conversation"]
        
        for msg in conversation_data:
            history.append(create_message_component(msg["content"], msg["role"]))
        
        # Success notification
        notification = html.Div(
            f"Successfully imported chat from {filename}",
            className="alert alert-success"
        )
        
        return history, conversation_data, notification
    except Exception as e:
        error_msg = f"Error importing chat: {str(e)}"
        print(error_msg)
        return dash.no_update, dash.no_update, html.Div(
            error_msg,
            className="alert alert-danger"
        )

# Trigger file selection dialog when import button is clicked
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            document.getElementById('upload-chat-json').click();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("import-chat-button", "n_clicks", allow_duplicate=True),
    Input("import-chat-button", "n_clicks"),
    prevent_initial_call=True
)

# Add keyboard events JavaScript for usability
app.clientside_callback(
    """
    function bindKeyEvents(n) {
        // Only set up listeners once
        if (window.keyBindingsSetup) return window.dash_clientside.no_update;
        
        // Function to toggle settings when Ctrl+, is pressed
        document.addEventListener('keydown', function(e) {
            // Ctrl+, to toggle settings
            if (e.ctrlKey && e.key === ',') {
                document.getElementById('settings-toggle').click();
                e.preventDefault();
            }
            
            // Ctrl+Enter to send message
            if (e.ctrlKey && e.key === 'Enter') {
                // Only if input is not empty
                const input = document.getElementById('message-input');
                if (input && input.value.trim()) {
                    document.getElementById('send-button').click();
                    e.preventDefault();
                }
            }
            
            // Escape to close settings if open
            if (e.key === 'Escape') {
                const settingsPanel = document.getElementById('settings-collapse');
                if (settingsPanel && settingsPanel.classList.contains('show')) {
                    document.getElementById('settings-toggle').click();
                    e.preventDefault();
                }
            }
            
            // Ctrl+S to export chat
            if (e.ctrlKey && e.key === 's') {
                document.getElementById('export-chat-button').click();
                e.preventDefault();
            }
            
            // Ctrl+O to import chat
            if (e.ctrlKey && e.key === 'o') {
                document.getElementById('import-chat-button').click();
                e.preventDefault();
            }
        });
        
        window.keyBindingsSetup = true;
        return window.dash_clientside.no_update;
    }
    """,
    Output("notification-area", "children", allow_duplicate=True),
    Input("chat-history", "children"),
    prevent_initial_call=True
)

# Search modal toggle
@callback(
    Output("search-modal", "is_open"),
    [Input("search-chat-button", "n_clicks"), Input("close-search-modal", "n_clicks")],
    [State("search-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_search_modal(search_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "search-chat-button":
        return True
    elif button_id == "close-search-modal":
        return False
    
    return is_open

# Search functionality
@callback(
    Output("search-results", "children"),
    [Input("search-input", "value")],
    [State("conversation-store", "data")],
    prevent_initial_call=True
)
def search_conversation(search_term, conversation_data):
    if not search_term or not conversation_data:
        return html.Div("Enter a search term to find similar messages")
    
    try:
        # Find similar messages using embeddings
        similar_messages = find_similar_messages(search_term, conversation_data, top_k=5)
        
        if not similar_messages:
            return html.Div("No similar messages found")
        
        # Display search results
        results = []
        for i, msg in enumerate(similar_messages):
            role_color = DARK_THEME["secondary"] if msg["role"] == "user" else DARK_THEME["primary"]
            results.append(html.Div([
                html.Div([
                    html.Strong(f"{msg['role'].capitalize()}: ", style={"color": role_color}),
                    html.Span(msg["content"][:150] + ("..." if len(msg["content"]) > 150 else ""))
                ]),
                html.Hr(style={"margin": "10px 0"}) if i < len(similar_messages) - 1 else None
            ], className="mb-3"))
        
        return html.Div(results)
    except Exception as e:
        return html.Div(f"Error searching: {str(e)}", className="text-danger")

# Prompt suggestions
@callback(
    Output("prompt-suggestions", "children"),
    [Input("chat-history", "children")],
    [State("conversation-store", "data")],
    prevent_initial_call=True
)
def update_prompt_suggestions(history, conversation_data):
    if not conversation_data or len(conversation_data) < 2:
        return html.Div()
    
    # Get the last assistant message
    assistant_messages = [msg for msg in conversation_data if msg["role"] == "assistant"]
    
    if not assistant_messages:
        return html.Div()
    
    last_assistant_message = assistant_messages[-1]["content"]
    
    # Generate suggestions
    suggestions = generate_prompt_suggestions(last_assistant_message)
    
    # Create suggestion pills
    suggestion_components = []
    for suggestion in suggestions:
        suggestion_components.append(
            dbc.Button(
                suggestion,
                id={"type": "suggestion-pill", "index": suggestions.index(suggestion)},
                color="light",
                size="sm",
                className="mr-2 mb-2",
                style={
                    "backgroundColor": "rgba(65, 105, 225, 0.1)",
                    "borderColor": "rgba(65, 105, 225, 0.3)",
                    "color": "white"
                }
            )
        )
    
    return html.Div([
        html.Small("Suggested follow-ups:", className="text-muted d-block mb-2"),
        html.Div(suggestion_components)
    ]) if suggestion_components else html.Div()

# Use suggestion pill
@callback(
    Output("message-input", "value", allow_duplicate=True),
    Input({"type": "suggestion-pill", "index": dash.ALL}, "n_clicks"),
    State({"type": "suggestion-pill", "index": dash.ALL}, "children"),
    prevent_initial_call=True
)
def use_suggestion(n_clicks, suggestions):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Find which button was clicked
    button_id = ctx.triggered[0]["prop_id"]
    index = json.loads(button_id.split(".")[0])["index"]
    
    # Return the clicked suggestion as the message input value
    return suggestions[index]

# Utility function for shortening long messages
def truncate_long_message(message, max_length=500):
    """Truncate long messages for display in UI components"""
    if len(message) <= max_length:
        return message
    
    # Truncate and add ellipsis
    return message[:max_length] + "..."

# Message similarity search using embeddings
def find_similar_messages(query, conversation_data, top_k=3):
    """Find messages in the conversation history similar to the query"""
    if not conversation_data or len(conversation_data) < 2:
        return []
    
    try:
        # Generate embedding for the query
        query_embedding = llm_client.get_embedding(query)
        
        # Generate embeddings for all messages in conversation history
        message_texts = [msg["content"] for msg in conversation_data]
        message_embeddings = llm_client.get_embedding(message_texts)
        
        # Calculate cosine similarity
        similarities = []
        for i, embedding in enumerate(message_embeddings):
            # Simple dot product for similarity (normalized embeddings)
            similarity = sum(q * e for q, e in zip(query_embedding, embedding))
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_similarities[:top_k]]
        
        # Return the most similar messages
        return [conversation_data[idx] for idx in top_indices]
    except Exception as e:
        print(f"Error finding similar messages: {str(e)}")
        return []

# Smart prompt suggestions based on message content
def generate_prompt_suggestions(message_text):
    """Generate smart suggestions for the next prompt based on the message content"""
    suggestions = []
    
    # Check for code-related content
    if "```" in message_text or "function" in message_text.lower() or "def " in message_text or "class " in message_text:
        suggestions.append("Can you explain how this code works?")
        suggestions.append("Can you optimize this code?")
        suggestions.append("What are potential bugs in this implementation?")
    
    # Check for explanations that might need clarification
    elif any(keyword in message_text.lower() for keyword in ["explain", "concept", "understanding", "means"]):
        suggestions.append("Can you provide a simpler explanation?")
        suggestions.append("Can you give me an example?")
        suggestions.append("How would you explain this to a beginner?")
    
    # Check for list-based responses
    elif any(pattern in message_text for pattern in ["1.", "2.", "•", "- ", "* "]):
        suggestions.append("Can you elaborate on point #1?")
        suggestions.append("Are there additional items you'd add to this list?")
        suggestions.append("Which of these points is most important?")
    
    # Check for comparison discussions
    elif any(keyword in message_text.lower() for keyword in ["versus", "compared to", "difference between", "pros and cons"]):
        suggestions.append("What's the key difference to remember?")
        suggestions.append("When would I choose one over the other?")
        suggestions.append("Can you summarize this comparison in a table?")
    
    # Check for error or problem discussions
    elif any(keyword in message_text.lower() for keyword in ["error", "problem", "issue", "bug", "fix"]):
        suggestions.append("What's the most common cause of this error?")
        suggestions.append("How can I prevent this problem?")
        suggestions.append("Is there a workaround?")
    
    # Default suggestions
    else:
        suggestions.append("Can you elaborate on that?")
        suggestions.append("How does this apply to real-world situations?")
        suggestions.append("What should I learn next about this topic?")
    
    # Return unique suggestions (removing duplicates if any)
    return list(dict.fromkeys(suggestions))

# Run the app
if __name__ == "__main__":
    # Create a cache directory if it doesn't exist
    Path("./cache").mkdir(exist_ok=True)
    
    # Create output directory for TTS files
    Path("./generated").mkdir(exist_ok=True)
    
    # Install required packages if not already installed
    try:
        import markdown
    except ImportError:
        import subprocess
        print("Installing required packages...")
        subprocess.check_call(["pip", "install", "markdown"])
    
    print("Starting AI Chat with TTS...")
    app.run_server(debug=True, port=8050)
