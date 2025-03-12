"""
Agentic Development System with Voice Narration (Bugfixes)
----------------------------------------------------------
A production-grade system combining LLM-powered code generation with MCTS and A* search.
This version fixes loops and streaming issues.
"""

import os
import time
import random
import sys
import shutil
import math
import heapq
import json
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, ConfigDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict

# Import your existing modules
from llm_module import LLMClient, Message
from tts_module import text_to_speech, TTSClient, GenerateAudioRequest


# =====================================================
# CORE DATA MODELS
# =====================================================

class ActionType(str, Enum):
    """Enumeration of possible development action types."""
    WRITE_CODE = "write_code"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENT = "document"
    PLAN = "plan"
    EVALUATE = "evaluate"


class DevAction(BaseModel):
    """Represents a development action that can be taken."""
    action_type: ActionType
    target: str = Field(..., description="Target of the action (e.g., file name, component)")
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(frozen=False)
    
    @model_validator(mode='before')
    @classmethod
    def validate_action(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure action is valid based on action_type."""
        action_type = data.get("action_type")
        content = data.get("content")
        
        if action_type in [ActionType.WRITE_CODE, ActionType.REFACTOR] and not content:
            raise ValueError(f"{action_type} actions must include content")
        
        return data
    
    def apply_to_state(self, state: 'DevState') -> 'DevState':
        """Apply this action to a state, returning a new state."""
        new_state = state.model_copy(deep=True)
        
        if self.action_type == ActionType.WRITE_CODE:
            new_state.code_artifacts[self.target] = self.content or ""
            # Mark related task as completed
            for task in new_state.tasks:
                if not task.completed and task.name in self.target:
                    new_state.completed_tasks.add(task.name)
                    task.completed = True
                    task.filename = self.target
                    task.code_artifact = self.content
                    break
        elif self.action_type == ActionType.REFACTOR:
            if self.target in new_state.code_artifacts and self.content:
                new_state.code_artifacts[self.target] = self.content
        elif self.action_type == ActionType.TEST:
            # In a real system, we'd run tests here. For now, just mark as tested.
            new_state.tested_files.add(self.target)
        elif self.action_type == ActionType.DOCUMENT:
            if self.target in new_state.code_artifacts and self.content:
                new_state.code_artifacts[self.target] = self.content
        
        return new_state


class DevTask(BaseModel):
    """Represents a development task to be completed."""
    name: str
    description: str
    priority: int = Field(1, ge=1)
    dependencies: List[str] = Field(default_factory=list)
    completed: bool = False
    code_artifact: Optional[str] = None
    filename: Optional[str] = None
    
    model_config = ConfigDict(frozen=False)
    
    def complete(self, filename: str, code_artifact: str):
        """Mark task as completed with its code artifact."""
        self.completed = True
        self.filename = filename
        self.code_artifact = code_artifact
    
    def __hash__(self):
        """Make hashable for use in sets."""
        return hash((self.name, self.description, self.priority))


class DevState(BaseModel):
    """Represents the current state of development."""
    project_name: str
    tasks: List[DevTask] = Field(default_factory=list)
    completed_tasks: Set[str] = Field(default_factory=set)
    code_artifacts: Dict[str, str] = Field(default_factory=dict)
    tested_files: Set[str] = Field(default_factory=set)
    evaluation_score: float = Field(0.0, ge=0.0, le=1.0)
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def get_completion_percentage(self) -> float:
        """Calculate the percentage of completed tasks."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for task in self.tasks if task.completed)
        return (completed / len(self.tasks)) * 100
    
    def get_next_task(self) -> Optional[DevTask]:
        """Get the highest priority incomplete task."""
        # Filter tasks that are incomplete
        incomplete_tasks = [t for t in self.tasks if not t.completed]
        
        if not incomplete_tasks:
            return None
            
        # Filter tasks whose dependencies are all met
        available_tasks = []
        for task in incomplete_tasks:
            deps_satisfied = all(
                dep in self.completed_tasks or 
                not any(t.name == dep and not t.completed for t in self.tasks)
                for dep in task.dependencies
            )
            if deps_satisfied:
                available_tasks.append(task)
                
        if not available_tasks:
            # If there are dependency cycles, just return the highest priority task
            return min(incomplete_tasks, key=lambda t: t.priority)
            
        # Return highest priority available task
        return min(available_tasks, key=lambda t: t.priority)
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.completed for task in self.tasks)
    
    def get_state_hash(self) -> str:
        """Generate a unique hash representing the current state."""
        completed_str = ",".join(sorted(self.completed_tasks))
        files_str = ",".join(sorted(self.code_artifacts.keys()))
        return f"{completed_str}|{files_str}"


# =====================================================
# UTILITY COMPONENTS
# =====================================================

class WorkspaceManager(BaseModel):
    """Manages the workspace directory for code generation."""
    project_name: str
    workspace_dir: Path = Field(default=None)
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.workspace_dir = self._create_workspace_dir()
    
    def _create_workspace_dir(self) -> Path:
        """Create and return the workspace directory."""
        # Clean project name for directory naming
        safe_name = "".join(c if c.isalnum() else "_" for c in self.project_name)
        workspace_path = Path("workspace") / safe_name
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_path = workspace_path / timestamp
        
        # Create directory if it doesn't exist
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created workspace directory: {workspace_path}")
        return workspace_path
    
    def save_file(self, filename: str, content: str) -> Path:
        """Save a file to the workspace directory."""
        # Ensure the filename is clean
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_filename.endswith(".py"):
            safe_filename += ".py"
            
        file_path = self.workspace_dir / safe_filename
        
        # Write the content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return file_path
    
    def save_development_summary(self, state: DevState) -> Path:
        """Save a development summary including all tasks and files."""
        summary_path = self.workspace_dir / "development_summary.md"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# {state.project_name} Development Summary\n\n")
            
            f.write("## Tasks\n\n")
            for i, task in enumerate(state.tasks, 1):
                status = "✅" if task.completed else "⬜"
                f.write(f"{i}. {status} **{task.name}**: {task.description}\n")
                if task.filename:
                    f.write(f"   - Implemented in: `{task.filename}`\n")
            
            f.write("\n## Generated Files\n\n")
            for filename in state.code_artifacts.keys():
                f.write(f"- [`{filename}`]({filename})\n")
                
        return summary_path
    
    def copy_source_modules(self) -> None:
        """Copy the source modules to the workspace for reference."""
        try:
            for module in ["llm_module.py", "tts_module.py"]:
                if os.path.exists(module):
                    shutil.copy2(module, self.workspace_dir / module)
                    print(f"📋 Copied {module} to workspace")
        except Exception as e:
            print(f"⚠️ Couldn't copy source modules: {e}")
    
    def get_workspace_path(self) -> str:
        """Get the absolute path to the workspace directory."""
        return str(self.workspace_dir.absolute())


class StreamPrinter(BaseModel):
    """Handles streaming output to console in a human-like way."""
    typing_speed: float = Field(0.005, description="Base delay between characters")
    variance: float = Field(0.005, description="Random variance in typing speed")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def print_stream(self, text: str, prefix: str = ""):
        """Print text with a typewriter effect."""
        if prefix:
            sys.stdout.write(prefix)
            sys.stdout.flush()
            
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            
            # Calculate delay - longer for punctuation
            delay = self.typing_speed
            if char in ['.', '!', '?', '\n']:
                delay *= 5
            elif char in [',', ';', ':']:
                delay *= 3
                
            # Add some randomness to make it feel more natural
            time.sleep(delay + random.uniform(0, self.variance))
            
        # Ensure there's a newline at the end
        if not text.endswith('\n'):
            sys.stdout.write('\n')
            sys.stdout.flush()


class VoiceNarrator(BaseModel):
    """Handles TTS narration of the development process."""
    voice: str = "af_bella"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    last_messages: Set[str] = Field(default_factory=set)
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def narrate(self, message: str, force: bool = False):
        """Convert text to speech and play it."""
        # Skip if this exact message was recently narrated
        if not force and message in self.last_messages:
            return
            
        # Keep track of recent messages (simple deduplication)
        self.last_messages.add(message)
        if len(self.last_messages) > 10:
            self.last_messages.pop()
            
        print(f"\n🔊 {message}")
        text_to_speech(
            text=message,
            voice=self.voice,
            speed=self.speed,
            auto_play=True
        )
    
    def announce_project_start(self, project_name: str):
        """Announce the start of a project."""
        self.narrate(f"Starting development of {project_name}.", force=True)
    
    def announce_planning(self):
        """Announce that planning is in progress."""
        self.narrate("Generating development plan.", force=True)
    
    def announce_task_start(self, task: DevTask):
        """Announce the start of a task."""
        self.narrate(f"Starting implementation of {task.name}.")
    
    def announce_task_completion(self, task: DevTask):
        """Announce the completion of a task."""
        self.narrate(f"Completed task: {task.name}.")
    
    def announce_milestone(self, completion_percentage: float):
        """Announce when a milestone is reached."""
        if completion_percentage % 25 == 0:
            self.narrate(f"Milestone reached: {int(completion_percentage)}% of tasks completed.")
    
    def announce_project_completion(self, project_name: str, workspace_path: str):
        """Announce the completion of the project."""
        self.narrate(f"Project {project_name} has been successfully completed! All code files have been saved to {workspace_path}.", force=True)


# =====================================================
# LLM INTEGRATION
# =====================================================

class LLMDeveloper(BaseModel):
    """Uses LLM to generate code and evaluate progress."""
    system_prompt: str = """You are an expert Python developer specializing in clean, modular code.
Your task is to generate production-ready Python code that is:
1. Fully functional and complete
2. Well-documented with docstrings and comments
3. Properly structured with appropriate error handling
4. Following PEP 8 style guidelines
5. Using type hints for better code clarity

When asked to implement code, you will:
1. Generate a complete, working implementation
2. Include all necessary imports
3. Add proper error handling
4. Include docstrings and comments
5. Use meaningful variable names
6. Structure the code logically
"""
    client: Any = None
    stream_printer: Optional[StreamPrinter] = None
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.client = LLMClient()
        if not self.stream_printer:
            self.stream_printer = StreamPrinter()
    
    def create_development_plan(self, project_description: str) -> List[DevTask]:
        """Generate a development plan from a project description."""
        json_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1},
                    "dependencies": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "default": []
                    }
                },
                "required": ["name", "description", "priority"]
            }
        }
        
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"""
            Create a sequential development plan for the following project:
            
            {project_description}
            
            Break it down into 5-20 tasks with clear names, descriptions, priority order (1 being highest),
            and dependencies (task names that must be completed before this task can start).
            
            Return ONLY a JSON array of tasks with 'name', 'description', 'priority', and 'dependencies' fields.
            """)
        ]
        
        try:
            # Start with a message about planning
            self.stream_printer.print_stream("Thinking about the development plan...", prefix="\n🧠 ")
            
            plan_json = self.client.structured_completion(
                messages=messages,
                json_schema={"name": "dev_plan", "schema": json_schema},
                temperature=0.7
            )
            
            # Parse the response - it might be a string or a dict
            if isinstance(plan_json, str):
                try:
                    plan_data = json.loads(plan_json)
                except json.JSONDecodeError:
                    # If the LLM returned invalid JSON, use a fallback plan
                    print("⚠️ LLM returned invalid JSON for plan. Using fallback plan.")
                    return self._get_fallback_plan()
            else:
                plan_data = plan_json
                
            # Convert to DevTask objects
            tasks = []
            for task_data in plan_data:
                # Ensure dependencies field exists
                if 'dependencies' not in task_data:
                    task_data['dependencies'] = []
                    
                task = DevTask(
                    name=task_data["name"],
                    description=task_data["description"],
                    priority=task_data["priority"],
                    dependencies=task_data["dependencies"]
                )
                tasks.append(task)
            
            # Sort by priority
            tasks.sort(key=lambda t: t.priority)
            return tasks
            
        except Exception as e:
            print(f"❌ Error creating development plan: {e}")
            return self._get_fallback_plan()
    
    def _get_fallback_plan(self) -> List[DevTask]:
        """Return a fallback development plan if LLM fails."""
        return [
            DevTask(name="Setup project structure", 
                   description="Create basic project files and directory structure", 
                   priority=1),
            DevTask(name="Implement core functionality", 
                   description="Implement the main features of the project", 
                   priority=2,
                   dependencies=["Setup project structure"]),
            DevTask(name="Add error handling", 
                   description="Implement proper error handling and edge cases", 
                   priority=3,
                   dependencies=["Implement core functionality"]),
            DevTask(name="Create documentation", 
                   description="Add comments and documentation", 
                   priority=4,
                   dependencies=["Implement core functionality"])
        ]
    
    def implement_task(self, task: DevTask, current_code: Dict[str, str]) -> Tuple[str, str]:
        """Generate code implementation for a task using streaming."""
        # Format existing code context
        code_context = "\n\n".join([
            f"File: {filename}\n```python\n{content}\n```" 
            for filename, content in current_code.items()
        ])
        
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"""
            Generate a complete Python implementation for the following task:
            
            Task Name: {task.name}
            Description: {task.description}
            
            Requirements:
            1. The code must be complete and runnable
            2. Include all necessary imports
            3. Add proper error handling
            4. Include docstrings and comments
            5. Use type hints
            6. Follow PEP 8 style guidelines
            
            Existing Project Files:
            {code_context or "No existing code yet."}
            
            Return your response in the following format:
            
            filename.py
            ```python
            # Your complete implementation here
            ```
            
            The filename should be a Python-safe name based on the task name.
            Do not include any explanation or notes - only the filename and implementation.
            """)
        ]
        
        try:
            print(f"\n📝 Generating code for: {task.name}")
            print("=" * 50)
            
            # Get non-streaming response
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=-1,
                stream=False
            )
            
            # Print the response with typewriter effect
            self.stream_printer.print_stream(response, prefix="")
            print("\n" + "=" * 50)
            
            # Extract filename and code
            lines = response.strip().split("\n")
            filename = lines[0].strip().replace("```python", "").replace("```", "")
            
            # Clean up the filename if it has extra markdown
            filename = filename.split('`')[-1] if '`' in filename else filename
            
            # Ensure we have a safe filename based on the task name if extraction failed
            if not filename or len(filename) < 3 or ' ' in filename:
                safe_name = task.name.lower().replace(' ', '_')
                filename = f"{safe_name}.py"
            
            # Rest is the code content, clean up markdown
            code_content = "\n".join(lines[1:]).strip()
            code_content = code_content.replace("```python", "").replace("```", "").strip()
            
            # If code extraction failed, generate simple placeholder
            if not code_content:
                raise ValueError("Failed to extract code content from LLM response")
            
            return filename, code_content
            
        except Exception as e:
            print(f"❌ Error in implementation: {e}")
            # Fallback filename based on task name
            filename = f"{task.name.lower().replace(' ', '_')}.py"
            return filename, f"""
# Implementation for {task.name}

def main():
    print("Implementing {task.name}")
    
    # TODO: Add actual implementation here
    
if __name__ == "__main__":
    main()
"""
    
    def summarize_implementation(self, task: DevTask, code: str) -> str:
        """Generate a concise summary of the implementation."""
        messages = [
            Message(role="system", content="You are a technical communicator who creates clear, concise summaries."),
            Message(role="user", content=f"""
            Summarize what the following code implementation does in 1-2 sentences.
            Focus on the high-level purpose, not the details.
            
            Task: {task.name}
            
            Code:
            ```python
            {code}
            ```
            
            Provide ONLY the summary, nothing else.
            """)
        ]
        
        try:
            # Print a thinking indicator
            self.stream_printer.print_stream("Summarizing implementation...", prefix="\n🧠 ")
            
            # Use non-streaming mode to avoid JSON parsing issues
            summary = self.client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=-1,
                stream=False
            )
            
            return summary.strip()
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Implemented {task.name}"
    
    def generate_actions(self, state: DevState, n: int = 3) -> List[DevAction]:
        """Generate possible next actions from a state."""
        next_task = state.get_next_task()
        if not next_task:
            return []
            
        actions = []
        
        # Basic action - write code for the task
        filename = f"{next_task.name.lower().replace(' ', '_')}.py"
        
        # Generate sample content based on task
        content = f"""
# Implementation for {next_task.name}

def main():
    print("Implementing {next_task.name}")
    
    # TODO: Add actual implementation here
    
if __name__ == "__main__":
    main()
"""
        
        # Add the write code action
        actions.append(DevAction(
            action_type=ActionType.WRITE_CODE,
            target=filename,
            content=content
        ))
        
        # Add some variation in actions if we have existing code
        if state.code_artifacts:
            # Maybe suggest a refactor of an existing file
            existing_files = list(state.code_artifacts.keys())
            if existing_files:
                existing_file = random.choice(existing_files)
                refactored_content = state.code_artifacts[existing_file] + f"\n\n# Refactored for {next_task.name}"
                
                actions.append(DevAction(
                    action_type=ActionType.REFACTOR,
                    target=existing_file,
                    content=refactored_content
                ))
                
                # Maybe suggest testing
                actions.append(DevAction(
                    action_type=ActionType.TEST,
                    target=existing_file
                ))
            
        return actions[:n]  # Return at most n actions
    
    def evaluate_state(self, state: DevState) -> float:
        """Evaluate the quality of the development state."""
        # In a real system, we'd use the LLM to evaluate.
        # For simplicity, we'll use a heuristic based on completion percentage.
        completion = state.get_completion_percentage() / 100.0
        
        # Add some randomness to simulate LLM evaluation variance
        quality_factor = random.uniform(0.7, 1.0)
        
        return completion * quality_factor


# =====================================================
# SEARCH ALGORITHM IMPLEMENTATIONS
# =====================================================

class MCTSNode(BaseModel):
    """Monte Carlo Tree Search node for development planning."""
    state: DevState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = Field(default_factory=list)
    visits: int = Field(0, ge=0)
    value: float = Field(0.0)
    action: Optional[DevAction] = None
    g_cost: float = Field(0.0, description="Cost from start to current node")
    h_cost: float = Field(0.0, description="Heuristic cost to goal")
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible child nodes have been expanded."""
        return self.state.is_complete() or not self.state.get_next_task()
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (end of development)."""
        return self.state.is_complete()
    
    def ucb_score(self, exploration_weight: float = 1.0) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_weight * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0.0
            
        return exploitation + exploration


class TranspositionEntry:
    """Entry in the transposition table."""
    def __init__(self, value: float, depth: int, visits: int, best_action: Optional[DevAction]):
        self.value = value
        self.depth = depth
        self.visits = visits
        self.best_action = best_action


class MCTSPlanner(BaseModel):
    """Monte Carlo Tree Search planner for development."""
    llm_developer: LLMDeveloper
    max_depth: int = Field(50, ge=1)
    max_iterations: int = Field(200, description="Maximum number of MCTS iterations")
    exploration_weight: float = Field(1.0, description="Exploration weight for UCB1")
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def select_action(self, state: DevState) -> Optional[DevAction]:
        """Select the best action from current state using MCTS."""
        # Create root node
        root = MCTSNode(state=state)
        
        # Run MCTS iterations
        for _ in range(self.max_iterations):
            # Selection and expansion
            node = self._select_and_expand(root)
            
            # Simulation
            reward = self._rollout(node.state)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Select best child of the root
        if not root.children:
            return None
            
        # Select child with highest value
        best_child = max(root.children, key=lambda n: n.value / max(n.visits, 1))
        return best_child.action
    
    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        # If we can expand this node, do it
        if not node.is_fully_expanded():
            return self._expand(node)
            
        # Otherwise, select a child recursively
        if not node.is_terminal():
            selected_node = max(node.children, key=lambda n: n.ucb_score(self.exploration_weight))
            return self._select_and_expand(selected_node)
            
        # Terminal node reached
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a child."""
        # Generate possible actions
        actions = self.llm_developer.generate_actions(node.state)
        
        # Filter out actions already tried
        existing_actions = [child.action for child in node.children]
        new_actions = [a for a in actions if a not in existing_actions]
        
        if not new_actions:
            # If no new actions, consider this node fully expanded
            return node
            
        # Choose a random new action
        action = random.choice(new_actions)
        
        # Apply action to get new state
        new_state = action.apply_to_state(node.state)
        
        # Create new node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action
        )
        
        # Add to children
        node.children.append(child)
        
        return child
    
    def _rollout(self, state: DevState) -> float:
        """Simulate a random playout from state."""
        current_state = state.model_copy(deep=True)
        depth = 0
        
        while not current_state.is_complete() and depth < self.max_depth:
            # Get possible actions
            actions = self.llm_developer.generate_actions(current_state)
            
            if not actions:
                break
                
            # Choose random action
            action = random.choice(actions)
            
            # Apply action
            current_state = action.apply_to_state(current_state)
            depth += 1
        
        # Evaluate final state
        score = current_state.get_completion_percentage() / 100.0
        
        # Extra reward for completion
        if current_state.is_complete():
            score += 1.0
            
        return score
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
            
    def plan_implementation(self, state: DevState) -> List[DevAction]:
        """Plan a sequence of actions using A* search."""
        root_node = MCTSNode(state=state)
        goal_state = None
        
        # Priority queue for open set
        open_set: List[Tuple[float, int, MCTSNode]] = [(0, 0, root_node)]
        counter = 0  # Tiebreaker for equal f-costs
        
        # Track visited states and their costs
        closed_set: Dict[str, float] = {}
        
        # Track path
        came_from: Dict[str, Tuple[str, DevAction]] = {}
        
        while open_set and len(closed_set) < self.max_depth:
            # Get node with lowest f-cost
            current_f, _, current_node = heapq.heappop(open_set)
            current_hash = current_node.state.get_state_hash()
            
            # Skip if we've seen this state with a better cost
            if current_hash in closed_set and closed_set[current_hash] < current_f:
                continue
                
            # Check if we've reached the goal
            if current_node.state.is_complete():
                goal_state = current_node
                break
                
            # Mark as visited
            closed_set[current_hash] = current_f
            
            # Generate possible actions
            actions = self.llm_developer.generate_actions(current_node.state)
            
            # Explore neighbors
            for action in actions:
                # Apply action to get new state
                new_state = action.apply_to_state(current_node.state)
                new_hash = new_state.get_state_hash()
                
                # Calculate costs
                g_cost = current_node.g_cost + self._action_cost(action)
                h_cost = self._heuristic(new_state)
                f_cost = g_cost + h_cost
                
                # Skip if we've seen this state with a better cost
                if new_hash in closed_set and closed_set[new_hash] <= f_cost:
                    continue
                    
                # Create new node
                new_node = MCTSNode(
                    state=new_state,
                    parent=current_node,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    action=action
                )
                
                # Add to open set
                counter += 1
                heapq.heappush(open_set, (f_cost, counter, new_node))
                
                # Track path
                came_from[new_hash] = (current_hash, action)
        
        # Reconstruct path if goal found
        if goal_state:
            return self._reconstruct_path(came_from, goal_state.state.get_state_hash())
        
        return []
    
    def _action_cost(self, action: DevAction) -> float:
        """Calculate the cost of an action."""
        # Base cost
        base_cost = 1.0
        
        # Adjust based on action type
        if action.action_type == ActionType.WRITE_CODE:
            # Writing new code is more expensive
            base_cost *= 2.0
        elif action.action_type == ActionType.REFACTOR:
            # Refactoring is moderately expensive
            base_cost *= 1.5
        elif action.action_type == ActionType.TEST:
            # Testing is relatively cheap
            base_cost *= 0.5
            
        return base_cost
    
    def _heuristic(self, state: DevState) -> float:
        """Estimate cost to goal. Must be admissible (never overestimate)."""
        # Simple heuristic based on remaining tasks
        incomplete_tasks = sum(1 for task in state.tasks if not task.completed)
        
        # Calculate basic heuristic with admissible multiplier
        h_value = incomplete_tasks * 0.5  # Use 0.5 to ensure we don't overestimate
        
        # Add code quality assessment to heuristic
        quality_penalty = 0.0
        for file_path, code in state.code_artifacts.items():
            # Could analyze code quality metrics here
            if file_path.endswith('.py'):  # Process Python files differently
                # Example: longer Python files might be more complex and need refactoring
                lines = code.count('\n')
                if lines > 400:  # Long files may need refactoring
                    quality_penalty += 0.05  # Small penalty to guide towards better structure
        
        # Ensure heuristic remains admissible
        return h_value + min(quality_penalty, 0.2)  # Cap additional penalty
    
    def _reconstruct_path(self, came_from: Dict[str, Tuple[str, DevAction]], 
                         goal_hash: str) -> List[DevAction]:
        """Reconstruct the path from start to goal."""
        path = []
        current_hash = goal_hash
        
        while current_hash in came_from:
            # Extract source hash and action from the path history
            source_hash, action = came_from[current_hash]
            # Add action to our path
            path.append(action)
            # Move to previous state
            current_hash = source_hash
        
        # Path is built backwards from goal to start, so reverse it
        return list(reversed(path))


class EnhancedMCTSPlanner(MCTSPlanner):
    """Enhanced MCTS with better heuristics and optimizations."""
    def __init__(self, **data):
        super().__init__(**data)
        self.landmarks: Dict[str, Set[str]] = {}
        self.state_cache: Dict[str, float] = {}
        self.quality_metrics = CodeQualityMetrics()
    
    def plan_implementation(self, state: DevState) -> List[DevAction]:
        """Plan a sequence of actions using A* search with enhanced heuristics."""
        # Compute landmarks for better planning
        self._compute_landmarks(state)
        
        # Call parent implementation with enhanced features
        return super().plan_implementation(state)
    
    def select_action(self, state: DevState) -> Optional[DevAction]:
        """Select the best action using enhanced MCTS."""
        # Compute landmarks if not already done
        if not self.landmarks:
            self._compute_landmarks(state)
            
        # Use parent MCTS implementation
        return super().select_action(state)
    
    def _compute_landmarks(self, state: DevState) -> None:
        """Compute landmark tasks that must be completed."""
        # Clear existing landmarks
        self.landmarks.clear()
        
        # Initialize landmarks for each task
        for task in state.tasks:
            self.landmarks[task.name] = set()
            # Direct dependencies
            self.landmarks[task.name].update(task.dependencies)
        
        # Process indirect dependencies
        for task in state.tasks:
            for dep in task.dependencies:
                if dep in self.landmarks:  # Check if dependency exists in landmarks
                    self.landmarks[task.name].update(self.landmarks.get(dep, set()))
    
    def _enhanced_heuristic(self, state: DevState) -> float:
        """Enhanced heuristic considering multiple factors."""
        # Cache check
        state_hash = state.get_state_hash()
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        
        # Ensure landmarks are computed
        if not self.landmarks:
            self._compute_landmarks(state)
        
        # Base completion cost
        completion_cost = 1.0 - state.get_completion_percentage() / 100.0
        
        # Landmark-based cost
        landmark_cost = 0.0
        for task in state.tasks:
            if not task.completed and task.name in self.landmarks:
                remaining_landmarks = len(self.landmarks[task.name] - state.completed_tasks)
                landmark_cost += 0.1 * remaining_landmarks
        
        # Code quality cost (simplify calculation to focus on important metrics)
        quality_cost = 0.0
        for file_path, code in state.code_artifacts.items():  # Use file_path instead of filename
            metrics = self.quality_metrics.evaluate(code)
            quality_cost += (1.0 - metrics.overall_score) * 0.2
            
            # Use file path information if needed for additional heuristics
            if file_path.endswith('.py'):
                # Python files might have different quality weightings
                pass
        
        total_cost = completion_cost + landmark_cost + quality_cost
        self.state_cache[state_hash] = total_cost
        return total_cost
        
    def _heuristic(self, state: DevState) -> float:
        """Override parent heuristic with enhanced version."""
        return self._enhanced_heuristic(state)


class AStarNode(BaseModel):
    """A* search node for finding optimal implementation paths."""
    state: DevState
    parent: Optional['AStarNode'] = None
    g_cost: float = Field(0.0, description="Cost from start to current node")
    h_cost: float = Field(0.0, description="Heuristic cost to goal")
    action: Optional[DevAction] = None
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)


class AStarPlanner(BaseModel):
    """A* search planner for optimal development paths."""
    llm_developer: LLMDeveloper
    max_depth: int = Field(50, ge=1)
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def plan_implementation(self, state: DevState) -> List[DevAction]:
        """Plan a sequence of actions using A* search."""
        # Create start node
        start_node = AStarNode(state=state)
        
        # Priority queue for open set
        open_set: List[Tuple[float, int, AStarNode]] = [(0, 0, start_node)]
        counter = 0  # Tiebreaker for equal f-costs
        
        # Track visited states and their costs
        closed_set: Dict[str, float] = {}
        
        # Track path
        came_from: Dict[str, Tuple[str, DevAction]] = {}
        
        while open_set and len(closed_set) < self.max_depth:
            # Get node with lowest f-cost
            current_f, _, current_node = heapq.heappop(open_set)
            current_hash = current_node.state.get_state_hash()
            
            # Skip if we've seen this state with a better cost
            if current_hash in closed_set and closed_set[current_hash] < current_f:
                continue
                
            # Check if we've reached the goal
            if current_node.state.is_complete():
                return self._reconstruct_path(came_from, current_hash)
                
            # Mark as visited
            closed_set[current_hash] = current_f
            
            # Generate possible actions
            actions = self.llm_developer.generate_actions(current_node.state)
            
            # Explore neighbors
            for action in actions:
                # Apply action to get new state
                new_state = action.apply_to_state(current_node.state)
                new_hash = new_state.get_state_hash()
                
                # Calculate costs
                g_cost = current_node.g_cost + self._action_cost(action)
                h_cost = self._heuristic(new_state)
                f_cost = g_cost + h_cost
                
                # Skip if we've seen this state with a better cost
                if new_hash in closed_set and closed_set[new_hash] <= f_cost:
                    continue
                    
                # Create new node
                new_node = AStarNode(
                    state=new_state,
                    parent=current_node,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    action=action
                )
                
                # Add to open set
                counter += 1
                heapq.heappush(open_set, (f_cost, counter, new_node))
                
                # Track path
                came_from[new_hash] = (current_hash, action)
        
        # No path found
        return []
    
    def _action_cost(self, action: DevAction) -> float:
        """Calculate the cost of an action."""
        # Base cost
        base_cost = 1.0
        
        # Adjust based on action type
        if action.action_type == ActionType.WRITE_CODE:
            # Writing new code is more expensive
            base_cost *= 2.0
        elif action.action_type == ActionType.REFACTOR:
            # Refactoring is moderately expensive
            base_cost *= 1.5
        elif action.action_type == ActionType.TEST:
            # Testing is relatively cheap
            base_cost *= 0.5
            
        return base_cost
    
    def _heuristic(self, state: DevState) -> float:
        """Estimate cost to goal. Must be admissible (never overestimate)."""
        # Simple heuristic based on remaining tasks
        incomplete_tasks = sum(1 for task in state.tasks if not task.completed)
        # Use a safe multiplier to ensure we don't overestimate
        return incomplete_tasks * 0.5
    
    def _reconstruct_path(self, came_from: Dict[str, Tuple[str, DevAction]], 
                         goal_hash: str) -> List[DevAction]:
        """Reconstruct the path from start to goal."""
        path = []
        current_hash = goal_hash
        
        while current_hash in came_from:
            # Extract source hash and action from the path history
            source_hash, action = came_from[current_hash]
            # Add action to our path
            path.append(action)
            # Move to previous state
            current_hash = source_hash
        
        # Path is built backwards from goal to start, so reverse it
        return list(reversed(path))


class EnhancedAStarPlanner(AStarPlanner):
    """Enhanced A* planner with better heuristics and optimizations."""
    def __init__(self, **data):
        super().__init__(**data)
        self.landmarks: Dict[str, Set[str]] = {}
        self.state_cache: Dict[str, float] = {}
        self.quality_metrics = CodeQualityMetrics()
    
    def plan_implementation(self, state: DevState) -> List[DevAction]:
        """Plan a sequence of actions using enhanced A* search."""
        # Compute landmarks for better planning
        self._compute_landmarks(state)
        
        # Clear cache
        self.state_cache.clear()
        
        # Call parent implementation with enhanced features
        return super().plan_implementation(state)
    
    def _compute_landmarks(self, state: DevState) -> None:
        """Compute landmark tasks that must be completed."""
        # Clear existing landmarks
        self.landmarks.clear()
        
        # Initialize landmarks for each task
        for task in state.tasks:
            self.landmarks[task.name] = set()
            # Direct dependencies
            self.landmarks[task.name].update(task.dependencies)
        
        # Process indirect dependencies
        for task in state.tasks:
            for dep in task.dependencies:
                if dep in self.landmarks:
                    self.landmarks[task.name].update(self.landmarks.get(dep, set()))
    
    def _enhanced_heuristic(self, state: DevState) -> float:
        """Enhanced heuristic considering multiple factors."""
        # Cache check
        state_hash = state.get_state_hash()
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        
        # Ensure landmarks are computed
        if not self.landmarks:
            self._compute_landmarks(state)
        
        # Base completion cost
        completion_cost = 1.0 - state.get_completion_percentage() / 100.0
        
        # Landmark-based cost
        landmark_cost = 0.0
        for task in state.tasks:
            if not task.completed and task.name in self.landmarks:
                remaining_landmarks = len(self.landmarks[task.name] - state.completed_tasks)
                landmark_cost += 0.1 * remaining_landmarks
        
        # Code quality cost (simplify calculation to focus on important metrics)
        quality_cost = 0.0
        for file_path, code in state.code_artifacts.items():  # Use file_path instead of filename
            metrics = self.quality_metrics.evaluate(code)
            quality_cost += (1.0 - metrics.overall_score) * 0.2
        
        # Calculate total cost and cache it
        total_cost = completion_cost + landmark_cost + quality_cost
        self.state_cache[state_hash] = total_cost
        return total_cost
    
    def _heuristic(self, state: DevState) -> float:
        """Override parent heuristic with enhanced version."""
        return self._enhanced_heuristic(state)


class CodeQualityMetrics:
    """Evaluates code quality for better state evaluation."""
    def evaluate(self, code: str) -> 'QualityMetrics':
        complexity = self._calculate_complexity(code)
        maintainability = self._evaluate_maintainability(code)
        test_coverage = self._estimate_test_coverage(code)
        
        return QualityMetrics(
            complexity=complexity,
            maintainability=maintainability,
            test_coverage=test_coverage
        )
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity."""
        # Count decision points (if, for, while, etc.)
        decision_points = len([line for line in code.split('\n') 
                            if any(kw in line for kw in ['if ', 'for ', 'while ', 'except', 'elif', 'case'])])
        return 1.0 / (1.0 + 0.1 * decision_points)
    
    def _evaluate_maintainability(self, code: str) -> float:
        """Evaluate code maintainability."""
        lines = code.split('\n')
        # Check docstrings, comments, and line length
        has_docstring = '"""' in code or "'''" in code
        comment_ratio = len([l for l in lines if l.strip().startswith('#')]) / max(len(lines), 1)
        long_lines = len([l for l in lines if len(l) > 80])
        
        score = 0.0
        score += 0.3 if has_docstring else 0.0
        score += 0.3 * comment_ratio
        score += 0.4 * (1.0 - long_lines / max(len(lines), 1))
        return score
    
    def _estimate_test_coverage(self, code: str) -> float:
        """Estimate potential test coverage."""
        functions = [l for l in code.split('\n') if l.strip().startswith('def ')]
        test_functions = [f for f in functions if 'test_' in f]
        return len(test_functions) / max(len(functions), 1)


@dataclass
class QualityMetrics:
    """Holds code quality metrics."""
    complexity: float
    maintainability: float
    test_coverage: float
    
    @property
    def overall_score(self) -> float:
        return (self.complexity * 0.3 + 
                self.maintainability * 0.4 + 
                self.test_coverage * 0.3)


# =====================================================
# MAIN SYSTEM
# =====================================================

class AgenticDevelopmentSystem(BaseModel):
    """Main system that ties everything together."""
    project_name: str
    project_description: str
    voice: str = "af_bella"
    voice_speed: float = Field(0.9, ge=0.5, le=2.0)
    typing_speed: float = Field(0.005, ge=0.001, le=0.05)
    use_astar_planning: bool = True
    use_mcts_actions: bool = True
    max_iterations: int = Field(1000, description="Maximum number of iterations to prevent infinite loops")
    
    # Components (initialized later)
    narrator: Optional[VoiceNarrator] = None
    developer: Optional[LLMDeveloper] = None
    mcts_planner: Optional[EnhancedMCTSPlanner] = None
    astar_planner: Optional[EnhancedAStarPlanner] = None
    workspace: Optional[WorkspaceManager] = None
    stream_printer: Optional[StreamPrinter] = None
    state: Optional[DevState] = None
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize helpers
        self.stream_printer = StreamPrinter(typing_speed=self.typing_speed)
        self.narrator = VoiceNarrator(voice=self.voice, speed=self.voice_speed)
        
        # Initialize LLM developer
        self.developer = LLMDeveloper(
            system_prompt=self.system_prompt,
            stream_printer=self.stream_printer
        )
        
        # Initialize state
        self.state = DevState(project_name=self.project_name)
        
        # Initialize workspace
        self.workspace = WorkspaceManager(project_name=self.project_name)
        
        # Initialize planners (note that these depend on the developer being initialized first)
        self.mcts_planner = EnhancedMCTSPlanner(llm_developer=self.developer)
        self.astar_planner = EnhancedAStarPlanner(llm_developer=self.developer)
    
    def plan(self) -> List[DevTask]:
        """Create a development plan for the project."""
        self.narrator.announce_project_start(self.project_name)
        self.narrator.announce_planning()
        
        # Generate development tasks using the LLM
        tasks = self.developer.create_development_plan(self.project_description)
        self.state.tasks = tasks
        
        # Narrate the plan
        self.narrator.narrate(f"Development plan created with {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            dependency_str = ""
            if task.dependencies:
                dependency_str = f" (depends on: {', '.join(task.dependencies)})"
                
            task_msg = f"Task {i}: {task.name} - {task.description}{dependency_str}"
            self.stream_printer.print_stream(task_msg, prefix=f"\n📌 ")
            time.sleep(0.5)  # Brief pause between task announcements
        
        return tasks
    
    def develop(self) -> Dict[str, str]:
        """Execute the development process using search algorithms."""
        print(f"Starting development of {self.project_name}")
        try:
            # First, create a plan if not already created
            if not self.state.tasks:
                self.plan()
            
            # Copy the source modules to the workspace
            self.workspace.copy_source_modules()
            
            # Get strategic plan with A* if enabled
            strategic_plan = []
            if self.use_astar_planning:
                strategic_plan = self.astar_planner.plan_implementation(self.state)
                
                if strategic_plan:
                    plan_summary = f"Strategic plan created with {len(strategic_plan)} steps."
                    self.stream_printer.print_stream(plan_summary, prefix="\n🗺️ ")
                    self.narrator.narrate(plan_summary)
                    
                    # Print plan steps
                    for i, action in enumerate(strategic_plan, 1):
                        step_msg = f"Step {i}: {action.action_type.value} for {action.target}"
                        self.stream_printer.print_stream(step_msg, prefix="  • ")
            
            previous_completion = 0
            plan_index = 0
            iterations = 0
            
            # Process tasks until complete or max iterations reached
            while not self.state.is_complete() and iterations < self.max_iterations:
                iterations += 1
                
                # Get next task
                next_task = self.state.get_next_task()
                if not next_task:
                    print("⚠️ No more tasks available")
                    break
                    
                # Announce task start (only once per task)
                self.narrator.announce_task_start(next_task)
                
                # Determine action to take
                action = None
                
                # If using strategic plan and we haven't exhausted it
                if self.use_astar_planning and plan_index < len(strategic_plan):
                    action = strategic_plan[plan_index]
                    plan_index += 1
                    self.stream_printer.print_stream(
                        f"Following strategic plan - step {plan_index}", 
                        prefix="\n🧭 "
                    )
                    
                # If using MCTS or no strategic plan action available
                elif self.use_mcts_actions:
                    action = self.mcts_planner.select_action(self.state)
                    
                # If no action determined, generate implementation directly
                if not action or action.action_type not in [ActionType.WRITE_CODE, ActionType.REFACTOR]:
                    # Generate implementation
                    filename, code = self.developer.implement_task(next_task, self.state.code_artifacts)
                    
                    # Create action for consistency
                    action = DevAction(
                        action_type=ActionType.WRITE_CODE,
                        target=filename,
                        content=code
                    )
                
                # Apply the action to get a new state
                new_state = action.apply_to_state(self.state)
                
                # Save file to workspace for code actions
                if action.action_type in [ActionType.WRITE_CODE, ActionType.REFACTOR] and action.content:
                    file_path = self.workspace.save_file(action.target, action.content)
                    self.stream_printer.print_stream(f"Saved to {file_path}", prefix="\n💾 ")
                    
                    # Get a summary of what was implemented
                    summary = self.developer.summarize_implementation(next_task, action.content)
                    self.narrator.narrate(summary)
                    
                    # Mark task as completed in the new state
                    for task in new_state.tasks:
                        if task.name == next_task.name:
                            task.complete(action.target, action.content)
                            new_state.completed_tasks.add(task.name)
                    
                    # Update our state
                    self.state = new_state
                    
                    # Announce task completion
                    self.narrator.announce_task_completion(next_task)
                
                # Check for milestones
                current_completion = self.state.get_completion_percentage()
                if int(current_completion) // 25 > int(previous_completion) // 25:
                    milestone_pct = int(current_completion) // 25 * 25
                    self.narrator.announce_milestone(milestone_pct)
                previous_completion = current_completion
                
                # Small delay between tasks
                time.sleep(1)
            
            # Check if we hit the iteration limit
            if iterations >= self.max_iterations and not self.state.is_complete():
                print(f"\n⚠️ Reached maximum iterations ({self.max_iterations})")
                self.narrator.narrate("Development stopped due to reaching the maximum number of iterations.")
            
            # Save development summary
            summary_path = self.workspace.save_development_summary(self.state)
            self.stream_printer.print_stream(f"Development summary saved to {summary_path}", prefix="\n📑 ")
            
            # Announce project completion or status
            if self.state.is_complete():
                self.narrator.announce_project_completion(self.project_name, self.workspace.get_workspace_path())
            else:
                completion_pct = int(self.state.get_completion_percentage())
                self.narrator.narrate(f"Project development paused at {completion_pct}% completion. Files saved to {self.workspace.get_workspace_path()}")
            
            return self.state.code_artifacts
        except Exception as e:
            print(f"Development failed: {e}")
            raise


def main():
    """Run the agentic development system with search algorithms."""
    print("🚀 Starting Agentic Development System with Voice Narration")
    print("📚 Uses Monte Carlo Tree Search (MCTS) and A* for planning")
    print("🎯 Implements fixes for loops and streaming issues")
    print("="*50)
    
    # Project details
    project_name = "TimeSeriesAnalysisTool"
    project_description = """
    Create a Python tool for time series analysis that provides:
    1. Data loading and preprocessing for time series data
    2. Visual exploration of time series (trends, seasonality, autocorrelation)
    3. Basic forecasting using exponential smoothing and ARIMA models
    4. Anomaly detection in time series data 
    5. Performance metrics calculation and model comparison
    
    The tool should be modular, well-documented and include examples.
    """
    
    # Create and run the system
    system = AgenticDevelopmentSystem(
        project_name=project_name,
        project_description=project_description,
        voice="af_bella",
        voice_speed=0.9,
        typing_speed=0.005,  # Adjust typing speed (lower = faster)
        use_astar_planning=True,
        use_mcts_actions=True,
        max_iterations=7  # Limit the number of iterations to prevent infinite loops
    )
    
    # Execute development process
    _ = system.develop()  # Use underscore for unused return value
    
    # Print the workspace location
    workspace_path = system.workspace.get_workspace_path()
    print("\n" + "="*50)
    print(f"📁 All code generated in: {workspace_path}")  # Using workspace_path variable in f-string
    print("="*50)


if __name__ == "__main__":
    main()