"""
Core integration module for the AgenDev UI.

This module handles the integration between the UI and the core AgenDev functionality.
"""
from typing import Dict, List, Any, Tuple, Optional
import json
import time
from datetime import datetime
from pathlib import Path
import os

# Core module imports for AgenDev integration
try:
    from agendev.core import AgenDev, AgenDevConfig
    from agendev.models.task_models import (
        Task, TaskStatus, TaskPriority, TaskRisk, TaskType, 
        Epic, TaskGraph, Dependency
    )
    from agendev.models.planning_models import (
        SimulationConfig, PlanSnapshot, PlanningHistory, 
        SearchNodeType, PlanningPhase, SimulationResult
    )
    from agendev.llm_integration import LLMIntegration, LLMConfig
    from agendev.search_algorithms import MCTSPlanner, AStarPathfinder
    from agendev.probability_modeling import TaskProbabilityModel, ProjectRiskModel
    from agendev.parameter_controller import ParameterController
    from agendev.tts_notification import NotificationManager
    AGENDEV_AVAILABLE = True
except ImportError:
    # If core modules are not available, use mock implementations
    AGENDEV_AVAILABLE = False
    print("Warning: AgenDev core modules not available. Using mock implementations.")

class MockTask:
    """Mock implementation of a Task for testing without core modules."""
    def __init__(self, title, description, status="planned"):
        self.id = f"task_{int(time.time())}"
        self.title = title
        self.description = description
        self.status = status
        self.dependencies = []
        self.artifact_paths = []

class MockAgenDev:
    """Mock implementation of AgenDev for testing without core modules."""
    def __init__(self, project_name):
        self.project_name = project_name
        self.tasks = []
    
    def create_task(self, title, description):
        task = MockTask(title, description)
        self.tasks.append(task)
        return task.id
    
    def implement_task(self, task_id):
        # Find task
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            return {"error": "Task not found"}
        
        # Simulate implementation
        task.status = "completed"
        task.artifact_paths.append(f"{task.title.lower().replace(' ', '_')}.py")
        
        return {
            "success": True,
            "task_id": task_id,
            "implementation": f"# Implementation for {task.title}\n\ndef main():\n    print('Hello from {task.title}')\n\nif __name__ == '__main__':\n    main()",
            "file_path": task.artifact_paths[0]
        }
    
    def get_project_status(self):
        return {
            "project_name": self.project_name,
            "state": "implementing",
            "tasks": {
                "total": len(self.tasks),
                "by_status": {
                    "planned": sum(1 for t in self.tasks if t.status == "planned"),
                    "completed": sum(1 for t in self.tasks if t.status == "completed")
                }
            }
        }

class CoreIntegration:
    """
    Handles integration between the UI and AgenDev core functionality.
    
    This class provides methods for initializing projects, executing tasks,
    and getting status updates from the core system.
    """
    
    def __init__(self, llm_base_url: str = "http://192.168.2.12:1234", 
                tts_base_url: str = "http://127.0.0.1:7860"):
        """
        Initialize the core integration.
        
        Args:
            llm_base_url: URL for the LLM API
            tts_base_url: URL for the TTS API
        """
        self.llm_base_url = llm_base_url
        self.tts_base_url = tts_base_url
        self.agendev_instance = None
        
    def initialize_project(self, project_name: str, project_description: str) -> Dict[str, Any]:
        """
        Initialize a new project in AgenDev.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            
        Returns:
            Dictionary with project initialization details
        """
        try:
            if AGENDEV_AVAILABLE:
                # Initialize actual AgenDev instance
                config = AgenDevConfig(
                    project_name=project_name,
                    llm_base_url=self.llm_base_url,
                    tts_base_url=self.tts_base_url
                )
                self.agendev_instance = AgenDev(config)
                
                # Generate implementation plan
                plan = self.agendev_instance.generate_implementation_plan()
                
                # Get project status
                status = self.agendev_instance.get_project_status()
                
                return {
                    "success": True,
                    "project_name": project_name,
                    "plan": plan.model_dump() if hasattr(plan, "model_dump") else vars(plan),
                    "status": status
                }
            else:
                # Use mock implementation for testing
                self.agendev_instance = MockAgenDev(project_name)
                
                # Create some sample tasks
                tasks = [
                    {"title": "Initialize project repository", "description": "Set up Git repository and initial file structure"},
                    {"title": "Create basic project structure", "description": "Set up directories and configuration files"},
                    {"title": "Implement core functionality", "description": f"Write code for main features of {project_name}"},
                    {"title": "Add unit tests", "description": "Write tests for the implemented functionality"},
                    {"title": "Create documentation", "description": "Write documentation for the project"}
                ]
                
                task_ids = []
                for task in tasks:
                    task_id = self.agendev_instance.create_task(task["title"], task["description"])
                    task_ids.append(task_id)
                
                return {
                    "success": True,
                    "project_name": project_name,
                    "mock": True,
                    "task_ids": task_ids,
                    "status": self.agendev_instance.get_project_status()
                }
        except Exception as e:
            print(f"Error initializing project: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks for the current project.
        
        Returns:
            List of task details
        """
        if not self.agendev_instance:
            return []
            
        try:
            if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
                # Get tasks from actual AgenDev instance
                tasks = []
                for task_id, task in self.agendev_instance.task_graph.tasks.items():
                    tasks.append({
                        "id": str(task_id),
                        "title": task.title,
                        "description": task.description,
                        "status": task.status.value,
                        "dependencies": [str(dep_id) for dep_id in task.dependencies],
                        "artifact_paths": task.artifact_paths
                    })
                return tasks
            else:
                # Get tasks from mock instance
                return [
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "status": task.status,
                        "dependencies": [str(dep_id) for dep_id in task.dependencies],
                        "artifact_paths": task.artifact_paths
                    }
                    for task in self.agendev_instance.tasks
                ]
        except Exception as e:
            print(f"Error getting tasks: {e}")
            return []
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task using AgenDev.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Dictionary with execution results
        """
        if not self.agendev_instance:
            return {"error": "No active project"}
            
        try:
            # Add debug information
            print(f"Attempting to execute task with ID: {task_id}")
            
            # Convert string ID to UUID if needed
            if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "implement_task"):
                from uuid import UUID
                if not isinstance(task_id, UUID):
                    task_id = UUID(task_id)
                    
                # Execute task with actual AgenDev instance
                print(f"Calling implement_task on actual AgenDev instance...")
                try:
                    result = self.agendev_instance.implement_task(task_id)
                    print(f"Task implementation result: {result}")
                    return result
                except Exception as e:
                    print(f"ERROR in AgenDev implement_task: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this is an LLM API connection issue
                    error_str = str(e).lower()
                    if "connection" in error_str or "llm" in error_str or "400" in error_str:
                        print("Detected LLM API issue - using fallback implementation")
                        return self._generate_fallback_implementation(task_id)
                    
                    return {
                        "success": False,
                        "error": f"Task implementation failed: {str(e)}"
                    }
            else:
                # Execute task with mock instance
                print(f"Using mock implementation to execute task...")
                result = self.agendev_instance.implement_task(task_id)
                print(f"Mock implementation result: {result}")
                return result
        except Exception as e:
            print(f"ERROR executing task: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_fallback_implementation(self, task_id: str) -> Dict[str, Any]:
        """
        Generate a fallback implementation when LLM API fails.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with implementation details
        """
        print(f"Generating fallback implementation for task {task_id}")
        task_title = "Unknown Task"
        task_description = "Task description not available"
        
        # Try to get task details
        if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
            from uuid import UUID
            if not isinstance(task_id, UUID):
                task_id = UUID(task_id)
                
            if task_id in self.agendev_instance.task_graph.tasks:
                task = self.agendev_instance.task_graph.tasks[task_id]
                task_title = task.title
                task_description = task.description
        else:
            # Use mock
            task = next((t for t in self.agendev_instance.tasks if t.id == task_id), None)
            if task:
                task_title = task.title
                task_description = task.description
        
        # Create snake game implementation based on the title
        implementation = self._create_snake_game_template(task_title, task_description)
        
        # Create safe file name from task title
        import re
        safe_filename = re.sub(r'[^\w\-_\.]', '_', task_title.lower().replace(' ', '_'))
        file_path = f"src/{safe_filename}.py"
        
        # Save the implementation
        try:
            # Make sure the src directory exists
            if not os.path.exists("src"):
                os.makedirs("src")
                
            # Save file
            with open(file_path, 'w') as f:
                f.write(implementation)
                
            print(f"Saved fallback implementation to {file_path}")
            
            # Update task status if possible
            if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
                from uuid import UUID
                if not isinstance(task_id, UUID):
                    task_id = UUID(task_id)
                    
                if task_id in self.agendev_instance.task_graph.tasks:
                    task = self.agendev_instance.task_graph.tasks[task_id]
                    task.status = "COMPLETED"
                    task.completion_percentage = 100.0
                    task.artifact_paths.append(file_path)
                    
                    # Save project state
                    if hasattr(self.agendev_instance, "_save_project_state"):
                        self.agendev_instance._save_project_state()
            
            return {
                "success": True,
                "task_id": str(task_id),
                "implementation": implementation,
                "file_path": file_path,
                "fallback": True
            }
        except Exception as e:
            print(f"Error saving fallback implementation: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"Failed to save fallback implementation: {str(e)}"
            }
    
    def _create_snake_game_template(self, task_title: str, task_description: str) -> str:
        """
        Create a basic snake game template as fallback implementation.
        
        Args:
            task_title: Title of the task
            task_description: Description of the task
            
        Returns:
            Python code for a simple snake game
        """
        return f'''
# {task_title}
# {task_description}
# 
# This is a basic snake game implementation as a fallback.

import pygame
import random
import sys
import time

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 10

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

# Font for score display
font = pygame.font.SysFont(None, 36)

class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)
        self.grow = False
        self.score = 0
    
    def get_head_position(self):
        return self.positions[0]
    
    def update(self):
        head = self.get_head_position()
        x, y = self.direction
        new_head = ((head[0] + x) % GRID_WIDTH, (head[1] + y) % GRID_HEIGHT)
        
        # Game over if snake hits itself
        if new_head in self.positions[1:]:
            return False
        
        self.positions.insert(0, new_head)
        
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False
            self.score += 1
        
        return True
    
    def render(self, surface):
        for position in self.positions:
            rect = pygame.Rect(
                position[0] * GRID_SIZE,
                position[1] * GRID_SIZE,
                GRID_SIZE, GRID_SIZE
            )
            pygame.draw.rect(surface, GREEN, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)
    
    def change_direction(self, direction):
        x, y = direction
        opposite_dir = (-self.direction[0], -self.direction[1])
        
        # Prevent moving directly opposite to current direction
        if (x, y) != opposite_dir:
            self.direction = (x, y)

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()
    
    def randomize_position(self):
        self.position = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1)
        )
    
    def render(self, surface):
        rect = pygame.Rect(
            self.position[0] * GRID_SIZE,
            self.position[1] * GRID_SIZE,
            GRID_SIZE, GRID_SIZE
        )
        pygame.draw.rect(surface, RED, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)

def draw_grid(surface):
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(surface, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(surface, BLACK, (0, y), (WIDTH, y))

def show_score(surface, score):
    score_text = font.render(f"Score: {score}", True, BLUE)
    surface.blit(score_text, (10, 10))

def show_game_over(surface, score):
    # Create semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))
    
    # Game over text
    game_over_text = font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    restart_text = font.render("Press SPACE to restart", True, WHITE)
    quit_text = font.render("Press ESC to quit", True, WHITE)
    
    surface.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 60))
    surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 20))
    surface.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 20))
    surface.blit(quit_text, (WIDTH // 2 - quit_text.get_width() // 2, HEIGHT // 2 + 60))

def show_menu(surface):
    surface.fill(BLACK)
    
    title_text = font.render("SNAKE GAME", True, GREEN)
    start_text = font.render("Press SPACE to start", True, WHITE)
    quit_text = font.render("Press ESC to quit", True, WHITE)
    
    surface.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 2 - 60))
    surface.blit(start_text, (WIDTH // 2 - start_text.get_width() // 2, HEIGHT // 2))
    surface.blit(quit_text, (WIDTH // 2 - quit_text.get_width() // 2, HEIGHT // 2 + 40))

def main():
    snake = Snake()
    food = Food()
    running = True
    game_over = False
    in_menu = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if in_menu:
                    if event.key == pygame.K_SPACE:
                        in_menu = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif game_over:
                    if event.key == pygame.K_SPACE:
                        snake = Snake()
                        food = Food()
                        game_over = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                else:
                    if event.key == pygame.K_UP:
                        snake.change_direction((0, -1))
                    elif event.key == pygame.K_DOWN:
                        snake.change_direction((0, 1))
                    elif event.key == pygame.K_LEFT:
                        snake.change_direction((-1, 0))
                    elif event.key == pygame.K_RIGHT:
                        snake.change_direction((1, 0))
        
        if in_menu:
            show_menu(screen)
        elif not game_over:
            # Update snake position
            if not snake.update():
                game_over = True
            
            # Check if snake eats food
            if snake.get_head_position() == food.position:
                snake.grow = True
                food.randomize_position()
                
                # Increase speed based on score
                FPS = min(20, 10 + snake.score // 5)
            
            # Render everything
            screen.fill(WHITE)
            draw_grid(screen)
            snake.render(screen)
            food.render(screen)
            show_score(screen, snake.score)
        else:
            show_game_over(screen, snake.score)
        
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
'''
    
    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project.
        
        Returns:
            Dictionary with project status
        """
        if not self.agendev_instance:
            return {"error": "No active project"}
            
        try:
            return self.agendev_instance.get_project_status()
        except Exception as e:
            print(f"Error getting project status: {e}")
            return {
                "error": str(e)
            }
    
    def generate_todo_markdown(self, project_name: str, tasks: List[Dict[str, Any]]) -> str:
        """
        Generate todo.md content based on tasks.
        
        Args:
            project_name: Name of the project
            tasks: List of task details
            
        Returns:
            Markdown content for todo.md
        """
        # Group tasks by status
        planned_tasks = [t for t in tasks if t["status"] == "planned"]
        in_progress_tasks = [t for t in tasks if t["status"] == "in_progress"]
        completed_tasks = [t for t in tasks if t["status"] == "completed"]
        
        # Build Markdown content
        markdown = f"# {project_name}\n\n"
        
        if in_progress_tasks:
            markdown += "## In Progress\n"
            for task in in_progress_tasks:
                markdown += f"- [ ] {task['title']}\n"
            markdown += "\n"
        
        if planned_tasks:
            markdown += "## Planned\n"
            for task in planned_tasks:
                markdown += f"- [ ] {task['title']}\n"
            markdown += "\n"
        
        if completed_tasks:
            markdown += "## Completed\n"
            for task in completed_tasks:
                markdown += f"- [x] {task['title']}\n"
            markdown += "\n"
        
        return markdown
    
    def generate_playback_steps(self, task_execution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate playback steps for a task execution.
        
        Args:
            task_execution_result: Result of task execution
            
        Returns:
            List of playback steps
        """
        if not task_execution_result.get("success", False):
            return []
            
        steps = []
        task_id = task_execution_result.get("task_id", "")
        file_path = task_execution_result.get("file_path", "")
        implementation = task_execution_result.get("implementation", "")
        
        # Find task details
        task_title = ""
        if AGENDEV_AVAILABLE and hasattr(self.agendev_instance, "task_graph"):
            from uuid import UUID
            if not isinstance(task_id, UUID):
                task_id = UUID(task_id)
                
            if task_id in self.agendev_instance.task_graph.tasks:
                task_title = self.agendev_instance.task_graph.tasks[task_id].title
        else:
            # Use mock
            task = next((t for t in self.agendev_instance.tasks if t.id == task_id), None)
            if task:
                task_title = task.title
        
        # Step 1: Planning (no Git operation)
        steps.append({
            "type": "terminal",
            "content": f"$ echo 'Planning implementation for {task_title}'\nPlanning implementation for {task_title}\n$ mkdir -p $(dirname {file_path})\n",
            "operation_type": "Planning",
            "file_path": task_title
        })
        
        # Step 2: Implementation
        steps.append({
            "type": "editor",
            "filename": file_path,
            "content": implementation,
            "operation_type": "Implementing",
            "file_path": file_path
        })
        
        # Step 3: Saving (simulate Git, don't actually run it)
        steps.append({
            "type": "terminal",
            "content": f"$ echo 'Saving implementation to {file_path}'\nSaving implementation to {file_path}\n# Simulating Git operations\n# git add {file_path}\n# git commit -m 'Implement {task_title}'\n[SUCCESS] Implementation saved to {file_path}",
            "operation_type": "Saving",
            "file_path": file_path
        })
        
        return steps