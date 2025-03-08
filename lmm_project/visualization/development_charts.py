# Empty placeholder files 

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import os

from lmm_project.core.mind import Mind
from lmm_project.core.exceptions import VisualizationError
from lmm_project.core.types import DevelopmentalStage

class DevelopmentCharts:
    """
    Charts for visualizing developmental progress
    
    This class provides specialized visualizations for tracking
    developmental progress over time and across modules.
    """
    def __init__(self, output_dir: str = "visualization/output"):
        """
        Initialize the development charts
        
        Parameters:
        output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib
        plt.style.use('ggplot')
        
        # Stage colors
        self.stage_colors = {
            "prenatal": "lightblue",
            "infant": "lightgreen",
            "child": "yellow",
            "adolescent": "orange",
            "adult": "red"
        }
        
    def plot_developmental_trajectory(self, state_history: List[Dict[str, Any]], save: bool = True) -> Optional[str]:
        """
        Plot the developmental trajectory over time
        
        Parameters:
        state_history: List of state dictionaries from the state manager
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract data
            timestamps = []
            ages = []
            stages = []
            
            for i, state in enumerate(state_history):
                # Use index as timestamp if not available
                if "last_updated" in state:
                    try:
                        timestamps.append(datetime.fromisoformat(state["last_updated"]))
                    except (ValueError, TypeError):
                        timestamps.append(i)
                else:
                    timestamps.append(i)
                    
                if "age" in state:
                    ages.append(state["age"])
                else:
                    ages.append(0.0)
                    
                if "developmental_stage" in state:
                    stages.append(state["developmental_stage"])
                else:
                    stages.append("unknown")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot age
            ax.plot(timestamps, ages, marker='o', label='Age')
            
            # Add stage transitions
            prev_stage = None
            stage_regions = []
            
            for i, stage in enumerate(stages):
                if stage != prev_stage:
                    stage_regions.append((i, stage))
                    prev_stage = stage
            
            # Color regions by stage
            for i in range(len(stage_regions)):
                start_idx = stage_regions[i][0]
                stage = stage_regions[i][1]
                
                # Determine end index
                if i < len(stage_regions) - 1:
                    end_idx = stage_regions[i+1][0]
                else:
                    end_idx = len(timestamps) - 1
                
                # Get color
                color = self.stage_colors.get(stage, "gray")
                
                # Add colored region
                if start_idx < end_idx:
                    ax.axvspan(timestamps[start_idx], timestamps[end_idx], 
                              alpha=0.2, color=color, label=f"Stage: {stage}" if i == 0 else "")
            
            # Add labels
            ax.set_xlabel('Time')
            ax.set_ylabel('Age')
            ax.set_title('Developmental Trajectory')
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Add grid
            ax.grid(True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"developmental_trajectory_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot developmental trajectory: {e}")
    
    def plot_module_development_heatmap(self, state_history: List[Dict[str, Any]], save: bool = True) -> Optional[str]:
        """
        Plot a heatmap of module development over time
        
        Parameters:
        state_history: List of state dictionaries from the state manager
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract module development data
            module_data = {}
            
            for state in state_history:
                if "module_development" in state:
                    for module, level in state["module_development"].items():
                        if module not in module_data:
                            module_data[module] = []
                        module_data[module].append(level)
            
            # Ensure all modules have the same number of data points
            max_length = max(len(data) for data in module_data.values())
            for module in module_data:
                if len(module_data[module]) < max_length:
                    # Pad with zeros
                    module_data[module] = module_data[module] + [0.0] * (max_length - len(module_data[module]))
            
            # Create data matrix
            modules = sorted(module_data.keys())
            data_matrix = np.array([module_data[module] for module in modules])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            im = ax.imshow(data_matrix, aspect='auto', cmap='viridis')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Development Level", rotation=-90, va="bottom")
            
            # Set labels
            ax.set_yticks(np.arange(len(modules)))
            ax.set_yticklabels(modules)
            ax.set_xlabel('State History Index')
            ax.set_title('Module Development Heatmap')
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"module_development_heatmap_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot module development heatmap: {e}")
    
    def plot_developmental_milestones(self, milestones: List[Dict[str, Any]], save: bool = True) -> Optional[str]:
        """
        Plot developmental milestones on a timeline
        
        Parameters:
        milestones: List of milestone dictionaries
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Extract data
            ages = []
            descriptions = []
            modules = []
            
            for milestone in milestones:
                if "age" in milestone:
                    ages.append(milestone["age"])
                else:
                    ages.append(0.0)
                    
                if "description" in milestone:
                    descriptions.append(milestone["description"])
                else:
                    descriptions.append("Unknown milestone")
                    
                if "module" in milestone:
                    modules.append(milestone["module"])
                else:
                    modules.append("unknown")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create unique colors for each module
            unique_modules = list(set(modules))
            module_colors = {}
            
            for i, module in enumerate(unique_modules):
                module_colors[module] = plt.cm.tab10(i % 10)
            
            # Plot milestones
            for i, (age, desc, module) in enumerate(zip(ages, descriptions, modules)):
                color = module_colors.get(module, "gray")
                ax.scatter(age, i, color=color, s=100, zorder=2)
                ax.text(age + 0.05, i, desc, va='center')
            
            # Add module legend
            for module, color in module_colors.items():
                ax.scatter([], [], color=color, label=module)
            
            # Set labels
            ax.set_xlabel('Age')
            ax.set_yticks([])
            ax.set_title('Developmental Milestones')
            
            # Add legend
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Add grid
            ax.grid(True, axis='x')
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"developmental_milestones_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot developmental milestones: {e}")
    
    def plot_stage_transition_probabilities(self, mind: Mind, save: bool = True) -> Optional[str]:
        """
        Plot the probability of transitioning to the next developmental stage
        
        Parameters:
        mind: The mind instance
        save: Whether to save the plot to a file
        
        Returns:
        Path to the saved file if save=True, None otherwise
        """
        try:
            # Get current stage and age
            current_stage = mind.developmental_stage
            current_age = mind.age
            
            # Define stage thresholds (from mind.py)
            stage_thresholds = {
                "prenatal": 0.1,
                "infant": 1.0,
                "child": 3.0,
                "adolescent": 6.0,
                "adult": float('inf')
            }
            
            # Calculate transition probability based on proximity to threshold
            transition_probs = {}
            
            for stage, threshold in stage_thresholds.items():
                if stage == current_stage:
                    # Current stage
                    next_stage_key = None
                    next_threshold = None
                    
                    # Find next stage
                    stages = list(stage_thresholds.keys())
                    current_idx = stages.index(current_stage)
                    
                    if current_idx < len(stages) - 1:
                        next_stage_key = stages[current_idx + 1]
                        next_threshold = stage_thresholds[next_stage_key]
                        
                        if next_threshold != float('inf'):
                            # Calculate probability based on progress toward next threshold
                            stage_start = threshold
                            stage_duration = next_threshold - stage_start
                            progress = (current_age - stage_start) / stage_duration
                            transition_probs[next_stage_key] = min(1.0, max(0.0, progress))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            stages = list(transition_probs.keys())
            probs = list(transition_probs.values())
            
            bars = ax.bar(stages, probs, color=[self.stage_colors.get(s, "gray") for s in stages])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Set labels
            ax.set_xlabel('Next Stage')
            ax.set_ylabel('Transition Probability')
            ax.set_title(f'Stage Transition Probabilities (Current: {current_stage}, Age: {current_age:.2f})')
            ax.set_ylim(0, 1.1)
            
            # Add grid
            ax.grid(True, axis='y')
            
            # Tight layout
            plt.tight_layout()
            
            # Save or show
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stage_transition_probabilities_{timestamp}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            raise VisualizationError(f"Failed to plot stage transition probabilities: {e}")
    
    def generate_development_report(self, mind: Mind, save: bool = True) -> Optional[str]:
        """
        Generate a comprehensive development report
        
        Parameters:
        mind: The mind instance
        save: Whether to save the report to a file
        
        Returns:
        Path to the saved file if save=True, report text otherwise
        """
        try:
            # Create report
            report = []
            report.append("=" * 60)
            report.append(f"DEVELOPMENTAL PROGRESS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 60)
            report.append("")
            
            # General info
            report.append(f"Age: {mind.age:.2f}")
            report.append(f"Developmental Stage: {mind.developmental_stage}")
            report.append("")
            
            # Stage information
            report.append("-" * 60)
            report.append("DEVELOPMENTAL STAGE INFORMATION")
            report.append("-" * 60)
            
            # Define stage thresholds (from mind.py)
            stage_thresholds = {
                "prenatal": 0.1,
                "infant": 1.0,
                "child": 3.0,
                "adolescent": 6.0,
                "adult": float('inf')
            }
            
            # Find current and next stage
            stages = list(stage_thresholds.keys())
            current_stage = mind.developmental_stage
            current_idx = stages.index(current_stage)
            
            # Current stage info
            report.append(f"Current Stage: {current_stage}")
            
            if current_idx < len(stages) - 1:
                next_stage = stages[current_idx + 1]
                next_threshold = stage_thresholds[next_stage]
                
                if next_threshold != float('inf'):
                    # Calculate progress toward next stage
                    current_threshold = stage_thresholds[current_stage]
                    stage_duration = next_threshold - current_threshold
                    progress = (mind.age - current_threshold) / stage_duration
                    
                    report.append(f"Progress toward {next_stage}: {progress:.2%}")
                    report.append(f"Estimated age for transition: {next_threshold}")
            
            report.append("")
            
            # Module development by category
            report.append("-" * 60)
            report.append("MODULE DEVELOPMENT BY CATEGORY")
            report.append("-" * 60)
            
            # Group modules by type
            module_categories = {
                "Perception": ["perception", "attention"],
                "Cognition": ["memory", "executive", "consciousness"],
                "Communication": ["language"],
                "Emotional": ["emotion", "self_regulation"],
                "Social": ["social", "identity", "belief"],
                "Creative": ["creativity", "temporal"],
                "Motivational": ["motivation", "learning"]
            }
            
            for category, module_types in module_categories.items():
                report.append(f"{category}:")
                
                # Get modules in this category
                category_modules = {name: module for name, module in mind.modules.items() 
                                   if name in module_types}
                
                if category_modules:
                    # Sort by development level
                    sorted_modules = sorted(
                        category_modules.items(),
                        key=lambda x: x[1].development_level,
                        reverse=True
                    )
                    
                    for name, module in sorted_modules:
                        report.append(f"  - {name}: {module.development_level:.2f}")
                else:
                    report.append("  No modules in this category")
                    
                report.append("")
            
            # Join report
            report_text = "\n".join(report)
            
            # Save or return
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"development_report_{timestamp}.txt"
                filepath = self.output_dir / filename
                
                with open(filepath, "w") as f:
                    f.write(report_text)
                    
                return str(filepath)
            else:
                return report_text
                
        except Exception as e:
            raise VisualizationError(f"Failed to generate development report: {e}") 
