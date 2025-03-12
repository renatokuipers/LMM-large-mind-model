# fs_utils.py
"""File system utilities for AgenDev."""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pickle
import hashlib

# Base workspace structure definition
WORKSPACE_STRUCTURE = {
    "src": "Generated source code",
    "artifacts": {
        "snapshots": "Code snapshots (local version control)",
        "audio": "Voice feedback and summaries",
        "models": "Context and memory models"
    },
    "planning": {
        "search_trees": "MCTS simulation data",
        "pathfinding": "A* planning results",
        "simulations": "Outcome probabilities"
    },
    "progress": {
        "history": "Historical snapshots"
    },
    "quality": {
        "tests": "Generated tests",
        "reviews": "Implementation reviews",
        "benchmarks": "Performance benchmarks"
    }
}

# Default files to create in the workspace
DEFAULT_FILES = {
    "planning/epics.json": {"epics": []},
    "planning/tasks.json": {"tasks": []},
    "planning/roadmap.md": "# AgenDev Implementation Roadmap\nThis document outlines the implementation timeline for the AgenDev project.\n",
    "progress/current.md": "# Current Project State\nThis document tracks the current state of the AgenDev project.\n",
    "progress/metrics.json": {"metrics": []}
}

def get_workspace_root() -> Path:
    """Get the root path of the workspace."""
    # Try to find the workspace directory by looking for markers
    try:
        # Start with current working directory
        current_dir = Path.cwd()
        current_dir = Path(os.path.normpath(str(current_dir)))
        
        # Look for workspace directory up to 3 levels up
        for _ in range(4):
            # Check if workspace directory exists at this level
            workspace_path = current_dir / "workspace"
            if workspace_path.exists() and workspace_path.is_dir():
                return workspace_path
                
            # Check if we're already in a workspace directory
            if current_dir.name == "workspace":
                return current_dir
                
            # Check if we're at root directory (can't go up further)
            if current_dir.parent == current_dir:
                break
                
            # Go up one level
            current_dir = current_dir.parent
        
        # If not found, check execution directory
        app_dir = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent.parent.parent
        workspace_path = app_dir / "workspace"
        if workspace_path.exists() and workspace_path.is_dir():
            return workspace_path
        
        # If still not found, create in current working directory
        workspace_path = Path.cwd() / "workspace"
        os.makedirs(workspace_path, exist_ok=True)
        print(f"Created workspace directory at: {workspace_path}")
        return workspace_path
        
    except Exception as e:
        print(f"Error locating workspace root: {e}")
        # Fallback to current directory + workspace
        workspace_path = Path.cwd() / "workspace"
        os.makedirs(workspace_path, exist_ok=True)
        return workspace_path

def ensure_workspace_structure() -> Path:
    """Create the workspace directory structure if it doesn't exist."""
    workspace_root = get_workspace_root()
    
    def create_nested_directories(parent_path: Path, structure: Union[Dict, str]) -> None:
        """Recursively create nested directories from the structure definition."""
        if isinstance(structure, dict):
            for dirname, substructure in structure.items():
                dir_path = parent_path / dirname
                os.makedirs(dir_path, exist_ok=True)
                
                # Create README.md with description
                if isinstance(substructure, str):
                    with open(dir_path / "README.md", "w") as f:
                        f.write(f"# {substructure}\n")
                
                # Process subdirectories
                if isinstance(substructure, dict):
                    create_nested_directories(dir_path, substructure)
        
    # Create the directory structure
    create_nested_directories(workspace_root, WORKSPACE_STRUCTURE)
    
    # Create default files
    for file_path, content in DEFAULT_FILES.items():
        full_path = workspace_root / file_path
        os.makedirs(full_path.parent, exist_ok=True)
        
        if isinstance(content, dict):
            save_json(content, full_path)
        else:
            with open(full_path, "w") as f:
                f.write(content)
    
    return workspace_root

def resolve_path(path: Union[str, Path], create_parents: bool = False) -> Path:
    """
    Resolve a path relative to the workspace root.
    
    Args:
        path: Path to resolve (absolute or relative to workspace)
        create_parents: Whether to create parent directories
        
    Returns:
        Absolute Path object
    """
    try:
        # Convert to Path if it's a string and normalize for Windows
        path_obj = Path(os.path.normpath(str(path))) if isinstance(path, str) else Path(os.path.normpath(str(path)))
        
        # If path is already absolute, use it directly
        if path_obj.is_absolute():
            if create_parents and path_obj.parent:
                os.makedirs(path_obj.parent, exist_ok=True)
            return path_obj
        
        # Otherwise, join with workspace root
        workspace_root = get_workspace_root()
        full_path = workspace_root / path_obj
        
        # Ensure the path is normalized (especially important for Windows)
        full_path = Path(os.path.normpath(str(full_path)))
        
        if create_parents and full_path.parent:
            try:
                os.makedirs(full_path.parent, exist_ok=True)
                print(f"Created directories for: {full_path.parent}")
            except Exception as e:
                print(f"Error creating directories for {full_path.parent}: {e}")
                
        return full_path
        
    except Exception as e:
        print(f"Error resolving path {path}: {e}")
        # Fall back to a safe option
        workspace_root = get_workspace_root()
        if isinstance(path, (str, Path)):
            filename = os.path.basename(str(path))
            safe_path = workspace_root / filename
            if create_parents:
                os.makedirs(safe_path.parent, exist_ok=True)
            return safe_path
        else:
            return workspace_root

def load_json(path: Union[str, Path]) -> Dict:
    """
    Load a JSON file.
    
    Args:
        path: Path to the JSON file, absolute or relative to workspace
        
    Returns:
        Parsed JSON data
    """
    full_path = resolve_path(path)
    
    if not full_path.exists():
        return {}
    
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If file exists but is not valid JSON, return empty dict
        return {}

def save_json(data: Dict, path: Union[str, Path], pretty: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save to, absolute or relative to workspace
        pretty: Whether to format the JSON for readability
    """
    full_path = resolve_path(path, create_parents=True)
    
    indent = 2 if pretty else None
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def safe_save_json(data: Dict, path: Union[str, Path], pretty: bool = True) -> bool:
    """
    Safely save data to a JSON file using a temporary file to prevent corruption.
    
    Args:
        data: Data to save
        path: Path to save to, absolute or relative to workspace
        pretty: Whether to format the JSON for readability
        
    Returns:
        True if save was successful, False otherwise
    """
    full_path = resolve_path(path, create_parents=True)
    
    # Create temporary file in the same directory
    temp_path = None
    try:
        # Create a temporary file in the same directory
        dir_path = full_path.parent
        os.makedirs(dir_path, exist_ok=True)
        
        # Use NamedTemporaryFile which properly handles cleanup
        with tempfile.NamedTemporaryFile(dir=dir_path, suffix=".json.tmp", delete=False, mode='w') as temp_file:
            temp_path = temp_file.name
            # Write data directly to the file object
            json.dump(data, temp_file, indent=2 if pretty else None, default=str)
            # Ensure file is flushed to disk
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        # Replace the original file with the temporary file (after temp_file is closed)
        shutil.move(temp_path, full_path)
        return True
        
    except Exception as e:
        print(f"Error saving JSON to {full_path}: {e}")
        # Attempt to clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def append_to_file(content: str, path: Union[str, Path], ensure_newline: bool = True) -> None:
    """
    Append content to a file.
    
    Args:
        content: Content to append
        path: Path to the file, absolute or relative to workspace
        ensure_newline: Whether to ensure the file ends with a newline
    """
    full_path = resolve_path(path, create_parents=True)
    
    # Ensure content ends with newline if requested
    if ensure_newline and not content.endswith('\n'):
        content += '\n'
    
    # Create the file if it doesn't exist
    if not full_path.exists():
        with open(full_path, 'w') as f:
            f.write(content)
        return
    
    # Check if the file ends with a newline
    needs_newline = False
    if ensure_newline:
        try:
            with open(full_path, 'r') as f:
                f.seek(max(0, os.path.getsize(full_path) - 1))
                last_char = f.read(1)
                needs_newline = last_char != '\n'
        except:
            # If an error occurs, assume no newline
            needs_newline = True
    
    # Append to the file
    with open(full_path, 'a') as f:
        if needs_newline:
            f.write('\n')
        f.write(content)

def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        path: Path to save to, absolute or relative to workspace
    """
    full_path = resolve_path(path, create_parents=True)
    
    with open(full_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: Union[str, Path], default=None) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        path: Path to the pickle file, absolute or relative to workspace
        default: Value to return if file doesn't exist
        
    Returns:
        Unpickled data or default value
    """
    full_path = resolve_path(path)
    
    if not full_path.exists():
        return default
    
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def content_hash(content: str) -> str:
    """
    Generate a hash of the content.
    
    Args:
        content: Content to hash
        
    Returns:
        SHA256 hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def save_snapshot(content: str, file_path: Union[str, Path], 
                 metadata: Optional[Dict] = None) -> Tuple[str, Path]:
    """
    Save a snapshot of content with metadata.
    
    Args:
        content: Content to save
        file_path: Original file path, used to determine snapshot location
        metadata: Additional metadata to store with the snapshot
        
    Returns:
        Tuple of (content_hash, snapshot_path)
    """
    # Convert the file path to a relative path if it's not already
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Normalize path for Windows
    file_path = Path(os.path.normpath(str(file_path)))
    
    if file_path.is_absolute():
        try:
            workspace_root = get_workspace_root()
            # Handle Windows paths
            workspace_root_str = str(workspace_root)
            file_path_str = str(file_path)
            
            # Ensure we handle Windows paths correctly
            if file_path_str.startswith(workspace_root_str):
                rel_path = Path(file_path_str[len(workspace_root_str):].lstrip('/\\'))
            else:
                # If the file is outside the workspace, use the filename only
                rel_path = Path(file_path.name)
        except ValueError as e:
            # If the file is outside the workspace, use the filename only
            print(f"Path resolution error: {e}")
            rel_path = Path(file_path.name)
    else:
        rel_path = file_path
    
    # Generate hash for the content
    hash_value = content_hash(content)
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    snapshot_metadata = {
        "hash": hash_value,
        "timestamp": datetime.now().isoformat(),
        "original_path": str(rel_path),
        **metadata
    }
    
    # Create parent directories safely by splitting into parts
    # First create the main snapshot directory
    snapshot_base = resolve_path("artifacts/snapshots", create_parents=True)
    
    # Then create the subdirectories based on rel_path.parent
    rel_parent = rel_path.parent
    if rel_parent and str(rel_parent) != '.':
        # Create each part of the directory path separately to avoid issues
        snapshot_dir = snapshot_base / rel_parent
        os.makedirs(snapshot_dir, exist_ok=True)
    else:
        snapshot_dir = snapshot_base
    
    # Generate unique filenames for content and metadata
    snapshot_stem = f"{rel_path.stem}_{hash_value[:8]}"
    snapshot_content_path = snapshot_dir / f"{snapshot_stem}{rel_path.suffix}"
    snapshot_metadata_path = snapshot_dir / f"{snapshot_stem}.meta.json"
    
    # Save content and metadata with improved error handling
    try:
        with open(snapshot_content_path, 'w') as f:
            f.write(content)
        
        save_json(snapshot_metadata, snapshot_metadata_path)
        print(f"Successfully saved snapshot to {snapshot_content_path}")
        return hash_value, snapshot_content_path
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        # Return the hash and an empty path as fallback
        return hash_value, Path("")

def list_snapshots(file_path: Union[str, Path]) -> List[Dict]:
    """
    List all snapshots for a file.
    
    Args:
        file_path: Original file path
        
    Returns:
        List of snapshot metadata dictionaries
    """
    # Convert the file path to a relative path if it's not already
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if file_path.is_absolute():
        try:
            workspace_root = get_workspace_root()
            rel_path = file_path.relative_to(workspace_root)
        except ValueError:
            # If the file is outside the workspace, use the filename only
            rel_path = Path(file_path.name)
    else:
        rel_path = file_path
    
    # Find all metadata files for this path
    snapshot_dir = resolve_path(f"artifacts/snapshots/{rel_path.parent}")
    if not snapshot_dir.exists():
        return []
    
    snapshots = []
    for meta_file in snapshot_dir.glob(f"{rel_path.stem}_*.meta.json"):
        try:
            metadata = load_json(meta_file)
            if metadata.get("original_path") == str(rel_path):
                snapshots.append(metadata)
        except:
            continue
    
    # Sort by timestamp
    return sorted(snapshots, key=lambda x: x.get("timestamp", ""), reverse=True)

def get_latest_snapshot(file_path: Union[str, Path]) -> Optional[Tuple[str, Dict]]:
    """
    Get the latest snapshot for a file.
    
    Args:
        file_path: Original file path
        
    Returns:
        Tuple of (content, metadata) or None if no snapshots exist
    """
    snapshots = list_snapshots(file_path)
    
    if not snapshots:
        return None
    
    latest = snapshots[0]
    hash_value = latest.get("hash", "")
    
    if not hash_value:
        return None
    
    # Convert the file path to a relative path
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    if file_path.is_absolute():
        try:
            workspace_root = get_workspace_root()
            rel_path = file_path.relative_to(workspace_root)
        except ValueError:
            rel_path = Path(file_path.name)
    else:
        rel_path = file_path
    
    # Construct the content path
    snapshot_dir = resolve_path(f"artifacts/snapshots/{rel_path.parent}")
    snapshot_content_path = snapshot_dir / f"{rel_path.stem}_{hash_value[:8]}{rel_path.suffix}"
    
    if not snapshot_content_path.exists():
        return None
    
    with open(snapshot_content_path, 'r') as f:
        content = f.read()
    
    return content, latest