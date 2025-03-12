"""File system utilities for AgenDev."""
import os
import json
import shutil
from typing import Dict, List, Optional, Set, Union, Any
from datetime import datetime
from pathlib import Path
import tempfile
from uuid import UUID


def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_workspace_path() -> Path:
    """
    Get the path to the AgenDev workspace.
    
    Returns:
        Path to the workspace directory
    """
    # Start with the current directory
    current_dir = Path.cwd()
    
    # Look for 'agendev/workspace' in the current or parent directories
    while current_dir != current_dir.parent:
        workspace_path = current_dir / 'agendev' / 'workspace'
        if workspace_path.exists():
            return workspace_path
        current_dir = current_dir.parent
    
    # If not found, use the default relative path
    return Path('agendev') / 'workspace'


def get_workspace_subdirectory(subdirectory: str) -> Path:
    """
    Get a subdirectory within the workspace and ensure it exists.
    
    Args:
        subdirectory: Name of the subdirectory
        
    Returns:
        Path to the subdirectory
    """
    workspace_path = get_workspace_path()
    subdirectory_path = workspace_path / subdirectory
    return ensure_directory(subdirectory_path)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> Path:
    """
    Save data as a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
        indent: Indentation level for the JSON file
        
    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    def json_default(obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with file_path.open('w') as f:
        json.dump(data, f, default=json_default, indent=indent)
    
    return file_path


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    with file_path.open('r') as f:
        return json.load(f)


def save_text(text: str, file_path: Union[str, Path]) -> Path:
    """
    Save text to a file.
    
    Args:
        text: Text to save
        file_path: Path to the file
        
    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with file_path.open('w') as f:
        f.write(text)
    
    return file_path


def load_text(file_path: Union[str, Path]) -> Optional[str]:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded text, or None if the file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    with file_path.open('r') as f:
        return f.read()


def save_binary(data: bytes, file_path: Union[str, Path]) -> Path:
    """
    Save binary data to a file.
    
    Args:
        data: Binary data to save
        file_path: Path to the file
        
    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with file_path.open('wb') as f:
        f.write(data)
    
    return file_path


def load_binary(file_path: Union[str, Path]) -> Optional[bytes]:
    """
    Load binary data from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded binary data, or None if the file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    with file_path.open('rb') as f:
        return f.read()


def create_temp_file(suffix: str = '', prefix: str = 'agendev_', content: Optional[Union[str, bytes]] = None) -> Path:
    """
    Create a temporary file.
    
    Args:
        suffix: Suffix for the temporary file
        prefix: Prefix for the temporary file
        content: Optional content to write to the file
        
    Returns:
        Path to the temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    
    temp_file_path = Path(temp_path)
    
    if content is not None:
        if isinstance(content, str):
            save_text(content, temp_file_path)
        else:
            save_binary(content, temp_file_path)
    
    return temp_file_path


def save_task_json(data: Any, task_id: Union[str, UUID]) -> Path:
    """
    Save task data to the tasks.json file.
    
    Args:
        data: Task data to save
        task_id: ID of the task
        
    Returns:
        Path to the saved file
    """
    planning_path = get_workspace_subdirectory('planning')
    tasks_file = planning_path / 'tasks.json'
    
    tasks_data = load_json(tasks_file) or {"tasks": []}
    
    # Check if the task already exists
    task_exists = False
    for i, task in enumerate(tasks_data['tasks']):
        if str(task.get('id', '')) == str(task_id):
            tasks_data['tasks'][i] = data
            task_exists = True
            break
    
    if not task_exists:
        tasks_data['tasks'].append(data)
    
    return save_json(tasks_data, tasks_file)


def save_epic_json(data: Any, epic_id: Union[str, UUID]) -> Path:
    """
    Save epic data to the epics.json file.
    
    Args:
        data: Epic data to save
        epic_id: ID of the epic
        
    Returns:
        Path to the saved file
    """
    planning_path = get_workspace_subdirectory('planning')
    epics_file = planning_path / 'epics.json'
    
    epics_data = load_json(epics_file) or {"epics": []}
    
    # Check if the epic already exists
    epic_exists = False
    for i, epic in enumerate(epics_data['epics']):
        if str(epic.get('id', '')) == str(epic_id):
            epics_data['epics'][i] = data
            epic_exists = True
            break
    
    if not epic_exists:
        epics_data['epics'].append(data)
    
    return save_json(epics_data, epics_file)


def get_source_directory() -> Path:
    """
    Get the path to the workspace source directory.
    
    Returns:
        Path to the source directory
    """
    return get_workspace_subdirectory('src')


def get_artifacts_directory() -> Path:
    """
    Get the path to the workspace artifacts directory.
    
    Returns:
        Path to the artifacts directory
    """
    return get_workspace_subdirectory('artifacts')


def get_audio_directory() -> Path:
    """
    Get the path to the workspace audio directory.
    
    Returns:
        Path to the audio directory
    """
    return get_workspace_subdirectory('artifacts/audio')


def get_snapshots_directory() -> Path:
    """
    Get the path to the workspace snapshots directory.
    
    Returns:
        Path to the snapshots directory
    """
    return get_workspace_subdirectory('artifacts/snapshots')


def get_models_directory() -> Path:
    """
    Get the path to the workspace models directory.
    
    Returns:
        Path to the models directory
    """
    return get_workspace_subdirectory('artifacts/models')


def get_tests_directory() -> Path:
    """
    Get the path to the workspace tests directory.
    
    Returns:
        Path to the tests directory
    """
    return get_workspace_subdirectory('quality/tests')


def get_planning_directory() -> Path:
    """
    Get the path to the workspace planning directory.
    
    Returns:
        Path to the planning directory
    """
    return get_workspace_subdirectory('planning')


def get_search_trees_directory() -> Path:
    """
    Get the path to the workspace search trees directory.
    
    Returns:
        Path to the search trees directory
    """
    return get_workspace_subdirectory('planning/search_trees')


def get_pathfinding_directory() -> Path:
    """
    Get the path to the workspace pathfinding directory.
    
    Returns:
        Path to the pathfinding directory
    """
    return get_workspace_subdirectory('planning/pathfinding')