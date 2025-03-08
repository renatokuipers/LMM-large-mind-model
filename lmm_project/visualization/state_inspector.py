# Empty placeholder files 

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import os

from lmm_project.core.mind import Mind
from lmm_project.core.exceptions import VisualizationError
from lmm_project.core.state_manager import StateManager

class StateInspector:
    """
    Inspector for visualizing and analyzing mind state
    
    This class provides methods for inspecting and visualizing
    the state of the mind, including state history, state changes,
    and state comparisons.
    """
    def __init__(self, output_dir: str = "visualization/output"):
        """
        Initialize the state inspector
        
        Parameters:
        output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib
        plt.style.use('ggplot')
        
    def plot_state_history(self, state_manager: StateManager, key: str, save: bool = True) -> Optional[str]:
        """
        Plot the history of a specific state value
        
        Parameters:
        state_manager: State manager instance
        key: State key to plot
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract data
            values = []
            timestamps = []
            
            for i, state in enumerate(state_manager.state_history):
                if key in state:
                    value = state[key]
                    
                    # Handle different value types
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, dict):
                        # For dictionaries, plot the number of items
                        values.append(len(value))
                    elif isinstance(value, list):
                        # For lists, plot the length
                        values.append(len(value))
                    else:
                        # For other types, just use 1 as a placeholder
                        values.append(1)
                        
                    # Use index as timestamp if not available
                    if "last_updated" in state:
                        try:
                            timestamps.append(datetime.fromisoformat(state["last_updated"]))
                        except (ValueError, TypeError):
                            timestamps.append(i)
                    else:
                        timestamps.append(i)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot values
            ax.plot(timestamps, values, marker='o')
            
            # Set labels
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f'State History: {key}')
            
            # Add grid
            ax.grid(True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_history_{key}_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot state history: {e}")
    
    def plot_state_changes(self, state_manager: StateManager, keys: List[str], save: bool = True) -> Optional[str]:
        """
        Plot changes in multiple state values over time
        
        Parameters:
        state_manager: State manager instance
        keys: List of state keys to plot
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract data
            data = {key: [] for key in keys}
            timestamps = []
            
            for i, state in enumerate(state_manager.state_history):
                # Use index as timestamp
                timestamps.append(i)
                
                # Extract values for each key
                for key in keys:
                    if key in state and isinstance(state[key], (int, float)):
                        data[key].append(state[key])
                    else:
                        # Use NaN for missing values
                        data[key].append(float('nan'))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot values for each key
            for key in keys:
                ax.plot(timestamps, data[key], marker='o', label=key)
            
            # Set labels
            ax.set_xlabel('State History Index')
            ax.set_ylabel('Value')
            ax.set_title('State Changes Over Time')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_changes_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot state changes: {e}")
    
    def compare_states(self, state1: Dict[str, Any], state2: Dict[str, Any], save: bool = True) -> Optional[str]:
        """
        Compare two states and visualize differences
        
        Parameters:
        state1: First state dictionary
        state2: Second state dictionary
        save: Whether to save the visualization to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Find common keys with numeric values
            common_keys = []
            for key in state1:
                if key in state2 and isinstance(state1[key], (int, float)) and isinstance(state2[key], (int, float)):
                    common_keys.append(key)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bar chart
            x = np.arange(len(common_keys))
            width = 0.35
            
            # Get values
            values1 = [state1[key] for key in common_keys]
            values2 = [state2[key] for key in common_keys]
            
            # Plot bars
            bars1 = ax.bar(x - width/2, values1, width, label='State 1')
            bars2 = ax.bar(x + width/2, values2, width, label='State 2')
            
            # Set labels
            ax.set_xlabel('State Key')
            ax.set_ylabel('Value')
            ax.set_title('State Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(common_keys, rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, axis='y')
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_comparison_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to compare states: {e}")
    
    def generate_state_diff_report(self, state1: Dict[str, Any], state2: Dict[str, Any], save: bool = True) -> Optional[str]:
        """
        Generate a text report of differences between two states
        
        Parameters:
        state1: First state dictionary
        state2: Second state dictionary
        save: Whether to save the report to a file
        
        Returns:
        Path to the saved file if save=True, report text otherwise
        """
        try:
            # Create report
            report = []
            report.append("=" * 60)
            report.append(f"STATE DIFFERENCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 60)
            report.append("")
            
            # Find all keys
            all_keys = set(state1.keys()) | set(state2.keys())
            
            # Added keys
            added_keys = set(state2.keys()) - set(state1.keys())
            if added_keys:
                report.append("-" * 60)
                report.append("ADDED KEYS")
                report.append("-" * 60)
                for key in sorted(added_keys):
                    report.append(f"{key}: {state2[key]}")
                report.append("")
            
            # Removed keys
            removed_keys = set(state1.keys()) - set(state2.keys())
            if removed_keys:
                report.append("-" * 60)
                report.append("REMOVED KEYS")
                report.append("-" * 60)
                for key in sorted(removed_keys):
                    report.append(f"{key}: {state1[key]}")
                report.append("")
            
            # Changed values
            changed_keys = []
            for key in set(state1.keys()) & set(state2.keys()):
                if state1[key] != state2[key]:
                    changed_keys.append(key)
            
            if changed_keys:
                report.append("-" * 60)
                report.append("CHANGED VALUES")
                report.append("-" * 60)
                for key in sorted(changed_keys):
                    report.append(f"{key}:")
                    report.append(f"  - Before: {state1[key]}")
                    report.append(f"  - After:  {state2[key]}")
                report.append("")
            
            # Join report
            report_text = "\n".join(report)
            
            # Save or return
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"state_diff_report_{timestamp}.txt"
                filepath = self.output_dir / filename
                
                with open(filepath, "w") as f:
                    f.write(report_text)
                    
                return str(filepath)
            else:
                return report_text
                
        except Exception as e:
            raise VisualizationError(f"Failed to generate state diff report: {e}")
    
    def inspect_mind_state(self, mind: Mind) -> Dict[str, str]:
        """
        Generate visualizations and reports for the mind's state
        
        Parameters:
        mind: The mind instance
        
        Returns:
        Dictionary mapping visualization types to file paths
        """
        results = {}
        
        # State history for age
        try:
            age_history_path = self.plot_state_history(mind.state_manager, "age")
            if age_history_path:
                results["age_history"] = age_history_path
        except Exception as e:
            print(f"Failed to plot age history: {e}")
        
        # State changes for module development
        if len(mind.state_manager.state_history) > 1:
            try:
                # Get module names from current state
                module_keys = []
                if "module_development" in mind.state_manager.current_state:
                    for module in mind.state_manager.current_state["module_development"]:
                        module_keys.append(f"module_development.{module}")
                
                if module_keys:
                    changes_path = self.plot_state_changes(mind.state_manager, module_keys[:5])  # Limit to 5 modules
                    if changes_path:
                        results["module_development_changes"] = changes_path
            except Exception as e:
                print(f"Failed to plot module development changes: {e}")
        
        # State comparison between first and current state
        if len(mind.state_manager.state_history) > 1:
            try:
                first_state = mind.state_manager.state_history[0]
                current_state = mind.state_manager.current_state
                
                comparison_path = self.compare_states(first_state, current_state)
                if comparison_path:
                    results["state_comparison"] = comparison_path
                    
                diff_report_path = self.generate_state_diff_report(first_state, current_state)
                if diff_report_path:
                    results["state_diff_report"] = diff_report_path
            except Exception as e:
                print(f"Failed to compare states: {e}")
        
        return results 
