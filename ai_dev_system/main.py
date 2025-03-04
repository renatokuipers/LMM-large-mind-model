import os
import sys
from pathlib import Path

# Ensure modules can be imported
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the Dash app
from ui.app import app
import ui.callbacks  # This needs to be imported to register callbacks

# Import models and services - these are imported in callbacks,
# but we import them here to ensure they're properly initialized
from models.project_model import ProjectStore
from llm_module import LLMClient
from services.planner import PlannerService
from services.code_generator import CodeGeneratorService

def main():
    """Main entry point for the application"""
    print("Starting LLM Full-Stack Developer System...")
    
    # Create necessary directories
    os.makedirs("projects", exist_ok=True)
    
    # Start the Dash app
    print("Starting Dash application...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()