# serialization.py - Utilities for saving and loading system state
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import gzip
from pydantic import BaseModel

logger = logging.getLogger("Serialization")

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_state(
    data: Any, 
    filepath: Union[str, Path], 
    compress: bool = False,
    use_pickle: bool = False
) -> bool:
    """Save state data to a file
    
    Args:
        data: Data to save (must be JSON serializable or Pydantic model)
        filepath: Path to save file
        compress: Whether to compress the data
        use_pickle: Whether to use pickle instead of JSON
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Convert Pydantic models to dict
        if isinstance(data, BaseModel):
            data = data.model_dump()
        
        # Choose serialization method
        if use_pickle:
            # Use pickle for complex Python objects
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
        else:
            # Use JSON for standard data
            if compress:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, cls=DateTimeEncoder, indent=2)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, cls=DateTimeEncoder, indent=2)
        
        logger.info(f"State saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving state to {filepath}: {str(e)}")
        return False

def load_state(
    filepath: Union[str, Path],
    default: Any = None,
    use_pickle: bool = False
) -> Any:
    """Load state data from a file
    
    Args:
        filepath: Path to load file from
        default: Value to return if loading fails
        use_pickle: Whether to use pickle instead of JSON
    
    Returns:
        Loaded data or default value if loading fails
    """
    try:
        filepath = Path(filepath)
        
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return default
        
        # Check if file is compressed
        is_compressed = False
        with open(filepath, 'rb') as f:
            magic_number = f.read(2)
            is_compressed = magic_number == b'\x1f\x8b'  # gzip magic number
        
        # Choose deserialization method
        if use_pickle:
            if is_compressed:
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
        else:
            if is_compressed:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        
        logger.info(f"State loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading state from {filepath}: {str(e)}")
        return default

def save_checkpoint(
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    metadata: Dict[str, Any],
    filepath: Union[str, Path]
) -> bool:
    """Save a training checkpoint
    
    Args:
        model_state: Model state dict
        optimizer_state: Optimizer state dict
        metadata: Additional metadata
        filepath: Path to save file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metadata": {
                **metadata,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Save with compression
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving checkpoint to {filepath}: {str(e)}")
        return False

def load_checkpoint(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load a training checkpoint
    
    Args:
        filepath: Path to load checkpoint from
    
    Returns:
        Checkpoint data or None if loading fails
    """
    try:
        filepath = Path(filepath)
        
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return None
        
        # Load with compression
        with gzip.open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error loading checkpoint from {filepath}: {str(e)}")
        return None

def export_data(
    data: Any,
    filepath: Union[str, Path],
    format: str = "json"
) -> bool:
    """Export data to a file in a specified format
    
    Args:
        data: Data to export
        filepath: Path to export file
        format: Export format (json, csv, pickle)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        
        # Convert Pydantic models to dict
        if isinstance(data, BaseModel):
            data = data.model_dump()
        
        # Export in specified format
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=DateTimeEncoder, indent=2)
        
        elif format.lower() == "csv":
            import csv
            # Handle different data structures
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # List of dictionaries - standard CSV export
                fieldnames = set()
                for item in data:
                    fieldnames.update(item.keys())
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
            elif isinstance(data, dict):
                # Dictionary - write key-value pairs
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Key", "Value"])
                    for key, value in data.items():
                        writer.writerow([key, value])
            else:
                raise ValueError(f"Cannot export data of type {type(data)} to CSV")
        
        elif format.lower() == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Data exported to {filepath} in {format} format")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data to {filepath}: {str(e)}")
        return False

def import_data(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Any:
    """Import data from a file
    
    Args:
        filepath: Path to import file from
        format: File format (if None, inferred from extension)
    
    Returns:
        Imported data or None if import fails
    """
    try:
        filepath = Path(filepath)
        
        if not os.path.exists(filepath):
            logger.error(f"Import file not found: {filepath}")
            return None
        
        # Infer format from extension if not specified
        if format is None:
            ext = filepath.suffix.lower()
            if ext == '.json':
                format = 'json'
            elif ext == '.csv':
                format = 'csv'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            else:
                raise ValueError(f"Cannot infer format from extension: {ext}")
        
        # Import based on format
        if format.lower() == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        elif format.lower() == "csv":
            import csv
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
        
        elif format.lower() == "pickle":
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        logger.info(f"Data imported from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error importing data from {filepath}: {str(e)}")
        return None