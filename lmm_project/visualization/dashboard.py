import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import os

from lmm_project.core.mind import Mind
from lmm_project.core.exceptions import VisualizationError

class Dashboard:
    """
    Dashboard for visualizing the mind's state
    
    This class provides methods for creating visualizations of the mind's
    state, including development progress, module activations, and more.
    """
    def __init__(self, output_dir: str = "visualization/output"):
        """
        Initialize the dashboard
        
        Parameters:
        output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib
        plt.style.use('ggplot')
        
    def plot_development_progress(self, mind: Mind, save: bool = True) -> Optional[str]:
        """
        Plot the development progress of all modules
        
        Parameters:
        mind: The mind instance
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Get module development levels
            modules = mind.modules
            module_names = list(modules.keys())
            development_levels = [module.development_level for module in modules.values()]
            
            # Sort by development level
            sorted_indices = np.argsort(development_levels)
            sorted_names = [module_names[i] for i in sorted_indices]
            sorted_levels = [development_levels[i] for i in sorted_indices]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_names, sorted_levels, color='skyblue')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', va='center')
            
            # Add title and labels
            ax.set_title(f'Module Development Levels (Age: {mind.age:.2f}, Stage: {mind.developmental_stage})')
            ax.set_xlabel('Development Level')
            ax.set_xlim(0, 1.1)
            
            # Add grid
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"development_progress_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot development progress: {e}")
    
    def plot_module_activations(self, mind: Mind, save: bool = True) -> Optional[str]:
        """
        Plot the activation levels of all modules
        
        Parameters:
        mind: The mind instance
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Get module activation levels
            modules = mind.modules
            module_names = list(modules.keys())
            activation_levels = [module.activation_level for module in modules.values()]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create radar chart
            num_modules = len(module_names)
            angles = np.linspace(0, 2*np.pi, num_modules, endpoint=False).tolist()
            
            # Close the plot
            activation_levels.append(activation_levels[0])
            angles.append(angles[0])
            module_names.append(module_names[0])
            
            # Plot
            ax.plot(angles, activation_levels, 'o-', linewidth=2)
            ax.fill(angles, activation_levels, alpha=0.25)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(module_names[:-1])
            
            # Set y limits
            ax.set_ylim(0, 1)
            
            # Add title
            ax.set_title(f'Module Activation Levels (Age: {mind.age:.2f}, Stage: {mind.developmental_stage})')
            
            # Make the plot circular
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Add grid
            ax.grid(True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"module_activations_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot module activations: {e}")
    
    def plot_development_over_time(self, state_history: List[Dict[str, Any]], save: bool = True) -> Optional[str]:
        """
        Plot the development progress over time
        
        Parameters:
        state_history: List of state dictionaries from the state manager
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract data
            ages = []
            stages = []
            module_development = {}
            
            for state in state_history:
                if "age" in state:
                    ages.append(state["age"])
                if "developmental_stage" in state:
                    stages.append(state["developmental_stage"])
                if "module_development" in state:
                    for module, level in state["module_development"].items():
                        if module not in module_development:
                            module_development[module] = []
                        module_development[module].append(level)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot age
            ax1.plot(ages, marker='o')
            ax1.set_ylabel('Age')
            ax1.set_title('Development Over Time')
            
            # Add stage transitions
            stage_changes = []
            for i in range(1, len(stages)):
                if stages[i] != stages[i-1]:
                    stage_changes.append(i)
                    ax1.axvline(x=i, color='r', linestyle='--', alpha=0.5)
            
            # Plot module development
            for module, levels in module_development.items():
                if len(levels) == len(ages):  # Ensure same length
                    ax2.plot(levels, label=module)
            
            ax2.set_xlabel('State History Index')
            ax2.set_ylabel('Development Level')
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Add grid
            ax1.grid(True)
            ax2.grid(True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"development_over_time_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot development over time: {e}")
    
    def generate_state_report(self, mind: Mind, save: bool = True) -> Optional[str]:
        """
        Generate a text report of the mind's state
        
        Parameters:
        mind: The mind instance
        save: Whether to save the report to a file
        
        Returns:
        Path to the saved file if save=True, report text otherwise
        """
        try:
            # Create report
            report = []
            report.append("=" * 50)
            report.append(f"MIND STATE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 50)
            report.append("")
            
            # General info
            report.append(f"Age: {mind.age:.2f}")
            report.append(f"Developmental Stage: {mind.developmental_stage}")
            report.append(f"Active Modules: {len(mind.modules)}")
            report.append("")
            
            # Module details
            report.append("-" * 50)
            report.append("MODULE DEVELOPMENT LEVELS")
            report.append("-" * 50)
            
            # Sort modules by development level
            sorted_modules = sorted(
                mind.modules.items(), 
                key=lambda x: x[1].development_level,
                reverse=True
            )
            
            for name, module in sorted_modules:
                report.append(f"{name}: {module.development_level:.2f} (activation: {module.activation_level:.2f})")
            
            report.append("")
            
            # State manager info
            report.append("-" * 50)
            report.append("STATE MANAGER")
            report.append("-" * 50)
            report.append(f"Last Updated: {mind.state_manager.last_updated}")
            report.append(f"History Size: {len(mind.state_manager.state_history)}")
            report.append("")
            
            # Event bus info
            report.append("-" * 50)
            report.append("EVENT BUS")
            report.append("-" * 50)
            report.append(f"Message History Size: {len(mind.event_bus.message_history)}")
            report.append(f"Subscriber Count: {sum(len(subs) for subs in mind.event_bus.subscribers.values())}")
            report.append("")
            
            # Join report
            report_text = "\n".join(report)
            
            # Save or return
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mind_report_{timestamp}.txt"
                filepath = self.output_dir / filename
                
                with open(filepath, "w") as f:
                    f.write(report_text)
                    
                return str(filepath)
            else:
                return report_text
                
        except Exception as e:
            raise VisualizationError(f"Failed to generate state report: {e}")
    
    def visualize_mind_state(self, mind: Mind) -> Dict[str, str]:
        """
        Generate a complete visualization of the mind's state
        
        Parameters:
        mind: The mind instance
        
        Returns:
        Dictionary mapping visualization types to file paths
        """
        results = {}
        
        # Development progress
        progress_path = self.plot_development_progress(mind)
        if progress_path:
            results["development_progress"] = progress_path
            
        # Module activations
        activations_path = self.plot_module_activations(mind)
        if activations_path:
            results["module_activations"] = activations_path
            
        # State report
        report_path = self.generate_state_report(mind)
        if report_path:
            results["state_report"] = report_path
            
        # Development over time (if history available)
        if mind.state_manager and len(mind.state_manager.state_history) > 1:
            history_path = self.plot_development_over_time(mind.state_manager.state_history)
            if history_path:
                results["development_over_time"] = history_path
                
        return results