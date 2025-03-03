@echo off
REM Creating project layout for llm-code-generator

REM Create root folder
mkdir llm-code-generator

REM Create root files with comments
echo # Modern dependency management > llm-code-generator\pyproject.toml
echo # Project documentation > llm-code-generator\README.md
echo # Main Dash application entry point > llm-code-generator\app.py
echo # Centralized configuration (LLM settings, paths, etc) > llm-code-generator\config.py

REM Create core folder and its files
mkdir llm-code-generator\core
type nul > llm-code-generator\core\__init__.py
echo # LLM integration and context management > llm-code-generator\core\llm_manager.py
echo # Signature-based code tracking system > llm-code-generator\core\code_memory.py
echo # Overall project orchestration > llm-code-generator\core\project_manager.py
echo # Code validation engine > llm-code-generator\core\validators.py
echo # Custom exception hierarchy > llm-code-generator\core\exceptions.py
echo # Code execution & testing utilities > llm-code-generator\core\execution.py

REM Create schemas folder and its files
mkdir llm-code-generator\schemas
type nul > llm-code-generator\schemas\__init__.py
echo # Models for functions, classes, methods > llm-code-generator\schemas\code_entities.py
echo # Project specification schemas > llm-code-generator\schemas\project_spec.py
echo # LLM input/output schemas > llm-code-generator\schemas\llm_io.py
echo # Task tracking & dependency models > llm-code-generator\schemas\generation_tasks.py
echo # Validation result schemas > llm-code-generator\schemas\validation.py

REM Create generators folder and its files
mkdir llm-code-generator\generators
type nul > llm-code-generator\generators\__init__.py
echo # Abstract base generator class > llm-code-generator\generators\base.py
echo # Creates project scaffolding > llm-code-generator\generators\project_initializer.py
echo # Database schemas & data models > llm-code-generator\generators\data_layer.py
echo # Service layer & business rules > llm-code-generator\generators\business_logic.py
echo # API endpoints & validation > llm-code-generator\generators\api_layer.py
echo # UI components & state management > llm-code-generator\generators\frontend.py
echo # Test suite generator > llm-code-generator\generators\tests_generator.py

REM Create templates folder and subfolders
mkdir llm-code-generator\templates
mkdir llm-code-generator\templates\python
mkdir llm-code-generator\templates\python\fastapi
mkdir llm-code-generator\templates\python\flask
mkdir llm-code-generator\templates\python\django
mkdir llm-code-generator\templates\nodejs
mkdir llm-code-generator\templates\nodejs\express
mkdir llm-code-generator\templates\nodejs\nestjs

REM Create prompts folder and its files
mkdir llm-code-generator\prompts
type nul > llm-code-generator\prompts\__init__.py
echo # Base system prompts > llm-code-generator\prompts\system_prompts.py
echo # System design prompts > llm-code-generator\prompts\architecture.py
echo # Code implementation prompts > llm-code-generator\prompts\implementation.py
echo # Code review prompts > llm-code-generator\prompts\validation.py
echo # Output schemas for structured generation > llm-code-generator\prompts\schemas.py

REM Create ui folder and its files
mkdir llm-code-generator\ui
type nul > llm-code-generator\ui\__init__.py
echo # Main app layout > llm-code-generator\ui\layout.py
echo # Dash callbacks > llm-code-generator\ui\callbacks.py
echo # UI state management > llm-code-generator\ui\state.py

REM Create ui\components subfolder and its files
mkdir llm-code-generator\ui\components
type nul > llm-code-generator\ui\components\__init__.py
echo # Project specification form > llm-code-generator\ui\components\project_form.py
echo # Generated code explorer > llm-code-generator\ui\components\code_explorer.py
echo # Real-time generation logs > llm-code-generator\ui\components\generation_log.py
echo # Project status display > llm-code-generator\ui\components\status_panel.py

REM Create utils folder and its files
mkdir llm-code-generator\utils
type nul > llm-code-generator\utils\__init__.py
echo # LLM context window utilities > llm-code-generator\utils\context_management.py
echo # Resolves component dependencies > llm-code-generator\utils\dependency_resolver.py
echo # Parses code to extract signatures > llm-code-generator\utils\code_parser.py
echo # Analyzes type compatibility > llm-code-generator\utils\type_analyzer.py
echo # Structured logging setup > llm-code-generator\utils\logging.py

REM Create tests folder and its files
mkdir llm-code-generator\tests
type nul > llm-code-generator\tests\__init__.py
echo # Test fixtures > llm-code-generator\tests\conftest.py

REM Create tests\test_core folder and its files
mkdir llm-code-generator\tests\test_core
type nul > llm-code-generator\tests\test_core\__init__.py
type nul > llm-code-generator\tests\test_core\test_llm_manager.py
type nul > llm-code-generator\tests\test_core\test_code_memory.py
type nul > llm-code-generator\tests\test_core\test_validators.py

REM Create tests\test_generators folder and its files
mkdir llm-code-generator\tests\test_generators
type nul > llm-code-generator\tests\test_generators\__init__.py
type nul > llm-code-generator\tests\test_generators\test_project_initializer.py
type nul > llm-code-generator\tests\test_generators\test_data_layer.py

REM Create tests\test_integration folder and its files
mkdir llm-code-generator\tests\test_integration
type nul > llm-code-generator\tests\test_integration\__init__.py
type nul > llm-code-generator\tests\test_integration\test_full_generation.py

REM Create generated folder and its files
mkdir llm-code-generator\generated
echo # Ignore generated projects in git > llm-code-generator\generated\.gitignore
echo # Instructions for generated projects > llm-code-generator\generated\README.md

echo Project layout created successfully.
pause
