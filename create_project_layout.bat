@echo off
echo Creating LLM Fullstack Generator Project...

:: Create root directory
mkdir llm_fullstack_generator
cd llm_fullstack_generator

:: Create main app file
echo # Main Dash application entry point > app.py

:: Create core directory and files
mkdir core
echo # Core package initialization > core\__init__.py
echo # Main sequential workflow coordinator > core\orchestrator.py
echo # Pydantic models for project state > core\project_context.py
echo # All data validation schemas > core\schemas.py

:: Create modules directory and subdirectories
mkdir modules

:: Planning module
mkdir modules\planning
echo # Generates structured EPIC tasks > modules\planning\epic_generator.py
echo # Reviews and refines tasks > modules\planning\task_validator.py

:: Generation module
mkdir modules\generation
echo # Single-component generator > modules\generation\code_generator.py
echo # Structured output templates > modules\generation\prompt_templates.py

:: Memory module
mkdir modules\memory
echo # Tracks code signatures > modules\memory\codebase_context.py

:: Validation module
mkdir modules\validation
echo # Validates generated components > modules\validation\code_validator.py

:: UI directory
mkdir ui
echo # Page layouts > ui\layouts.py
echo # Linear callback chain > ui\callbacks.py
mkdir ui\components

:: Utils directory
mkdir utils
echo # LLM interaction wrapper > utils\llm_client.py
echo # File system operations > utils\file_manager.py

:: Templates directory
mkdir templates
mkdir templates\project_structures
mkdir templates\component_templates

echo Project structure created successfully!
cd ..
echo You can now find the project in the llm_fullstack_generator directory.