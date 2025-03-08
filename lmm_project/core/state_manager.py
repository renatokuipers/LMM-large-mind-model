from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
from pathlib import Path

class StateManager(BaseModel):
    """
    Manages the state of the cognitive system
    
    The StateManager tracks the current state of the mind and its modules,
    provides methods to update the state, and handles state persistence.
    """
    current_state: Dict[str, Any] = Field(default_factory=dict)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history_size: int = Field(default=100)
    last_updated: datetime = Field(default_factory=datetime.now)
    save_directory: Path = Field(default=Path("storage/states"))
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def update_state(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current state with new information
        
        Parameters:
        state_update: Dictionary containing state updates
        
        Returns:
        Updated complete state
        """
        # Update current state
        self.current_state.update(state_update)
        
        # Record timestamp
        self.last_updated = datetime.now()
        self.current_state["last_updated"] = self.last_updated.isoformat()
        
        # Add to history
        self.state_history.append(self.current_state.copy())
        
        # Trim history if needed
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
            
        return self.current_state
    
    def get_state(self, key: Optional[str] = None) -> Any:
        """
        Get current state or a specific state value
        
        Parameters:
        key: Optional key to retrieve specific state value
        
        Returns:
        Current state or specific state value
        """
        if key:
            return self.current_state.get(key)
        
        return self.current_state
    
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get state history
        
        Parameters:
        limit: Maximum number of historical states to return
        
        Returns:
        List of historical states, most recent first
        """
        return self.state_history[-limit:]
    
    def save_state(self, filename: Optional[str] = None) -> Path:
        """
        Save current state to file
        
        Parameters:
        filename: Optional filename to save state to
        
        Returns:
        Path to saved state file
        """
        # Create directory if it doesn't exist
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage = self.current_state.get("developmental_stage", "unknown")
            filename = f"mind_state_{stage}_{timestamp}.json"
            
        # Save to file
        file_path = self.save_directory / filename
        with open(file_path, "w") as f:
            json.dump(self.current_state, f, indent=2, default=str)
            
        return file_path
    
    def load_state(self, filepath: str) -> Dict[str, Any]:
        """
        Load state from file
        
        Parameters:
        filepath: Path to state file
        
        Returns:
        Loaded state
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
            
        with open(file_path, "r") as f:
            loaded_state = json.load(f)
            
        # Update current state
        self.current_state = loaded_state
        
        # Add to history
        self.state_history.append(self.current_state.copy())
        
        # Trim history if needed
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
            
        return self.current_state
