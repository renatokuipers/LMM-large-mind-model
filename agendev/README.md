# AgenDev: Agentive Development Framework

AgenDev is a powerful framework for agentive software development, leveraging multiple specialized agents to automate the entire development lifecycle from planning to deployment. It's designed to work with local LLM models and prioritizes both Python and JavaScript projects.

## Features

- **Multi-Agent Architecture**: Specialized agents for planning, coding, integration, deployment, knowledge, and web automation
- **End-to-End Development**: Handles the complete development lifecycle
- **API-First Design**: RESTful API and WebSocket interfaces for integration
- **Extensible Framework**: Easy to add new agents and capabilities
- **Local LLM Integration**: Works with your local LLM models instead of relying on external APIs
- **Voice Notifications**: Optional TTS integration for voice feedback during development
- **Python & JavaScript Support**: Build both Python and JavaScript projects with equal ease

## Agents

AgenDev includes the following specialized agents:

1. **PlannerAgent**: Analyzes requirements and creates implementation plans
2. **CodeAgent**: Implements code based on plans and requirements
3. **IntegrationAgent**: Ensures proper integration between components
4. **DeploymentAgent**: Handles deployment to various platforms
5. **KnowledgeAgent**: Provides information and best practices
6. **WebAutomationAgent**: Automates web-based tasks like GitHub repository setup

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agendev.git
cd agendev

# Install the package
pip install -e .
```

## Configuration

AgenDev can be configured using a configuration file or environment variables:

### Configuration File

Create a `config.json` file in one of these locations:
- Current directory
- `~/.agendev/config.json`
- `/etc/agendev/config.json`

Example configuration:

```json
{
  "workspace_path": "/path/to/workspace",
  "host": "127.0.0.1",
  "port": 8000,
  "llm": {
    "provider": "local",
    "model": "qwen2.5-7b-instruct",
    "endpoint": "http://192.168.2.12:1234"
  }
}
```

### Environment Variables

You can also use environment variables:

```bash
# Server configuration
export AGENDEV_HOST=127.0.0.1
export AGENDEV_PORT=8000
export AGENDEV_WORKSPACE_PATH=/path/to/workspace

# LLM configuration
export AGENDEV_LLM_PROVIDER=local
export AGENDEV_LLM_MODEL=qwen2.5-7b-instruct
export AGENDEV_LLM_ENDPOINT=http://192.168.2.12:1234
```

## Local LLM Setup

AgenDev is designed to work with local LLM models. It integrates with your locally running LLM server via the provided endpoint. 

The default configuration assumes:
- Endpoint: http://192.168.2.12:1234
- Model: qwen2.5-7b-instruct
- Embedding model: text-embedding-nomic-embed-text-v1.5@q4_k_m

Make sure your local LLM server is running and accessible before starting AgenDev.

## Usage

### Starting the Server

```bash
# Start the AgenDev server
agendev
```

### API Endpoints

- `POST /api/project`: Create or modify a project
- `GET /api/projects`: List all projects
- `GET /api/project/{project_id}`: Get project details
- `GET /api/health`: Health check endpoint
- `POST /api/notify`: Send voice notifications (if TTS is configured)
- `WebSocket /ws`: Real-time communication

### Example: Creating a Python Project

```python
import requests

response = requests.post("http://localhost:8000/api/project", json={
    "type": "create_project",
    "project_name": "data-analyzer",
    "description": "A Python data analysis tool",
    "prompt": "Create a data analysis tool with Python, pandas, and matplotlib that can load CSV files, perform basic statistical analysis, and generate visualizations"
})

print(response.json())
```

### Example: Creating a JavaScript Project

```python
import requests

response = requests.post("http://localhost:8000/api/project", json={
    "type": "create_project",
    "project_name": "todo-app",
    "description": "A simple todo list application",
    "prompt": "Create a todo list application with React frontend and Express backend"
})

print(response.json())
```

### Example: WebSocket Communication

```python
import asyncio
import websockets
import json

async def connect():
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        # Create a project
        await websocket.send(json.dumps({
            "type": "create_project",
            "request_id": "123",
            "project_name": "data-analyzer",
            "description": "A Python data analysis tool",
            "prompt": "Create a data analysis tool with Python, pandas, and matplotlib"
        }))
        
        # Receive the response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(connect())
```

## Voice Notifications

If you have the TTS system configured, AgenDev can provide voice notifications about the development process:

```python
import requests

response = requests.post("http://localhost:8000/api/notify", json={
    "message": "Project setup completed successfully",
    "type": "success",
    "priority": "medium",
    "auto_play": True
})

print(response.json())
```

## Development

### Project Structure

```
agendev/
├── src/
│   └── agendev/
│       ├── agents/
│       │   ├── agent_base.py
│       │   ├── planner_agent.py
│       │   ├── code_agent.py
│       │   ├── integration_agent.py
│       │   ├── deployment_agent.py
│       │   ├── knowledge_agent.py
│       │   └── web_automation_agent.py
│       ├── utils/
│       │   ├── config.py
│       │   └── llm.py
│       ├── llm_module.py
│       ├── llm_integration.py
│       ├── tts_module.py
│       ├── tts_notification.py
│       ├── agenflow_manager.py
│       └── app.py
├── setup.py
└── README.md
```

### Running Tests

```bash
# Run tests
pytest
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
