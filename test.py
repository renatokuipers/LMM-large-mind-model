# test.py
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
import threading
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, ConfigDict


# Import your existing modules
from llm_module import LLMClient, Message
from tts_module import text_to_speech, TTSClient, GenerateAudioRequest, get_available_voices, play_audio



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
        
        # Write the content to the file with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return file_path
    
    def save_development_summary(self, state: DevState) -> Path:
        """Save a development summary including all tasks and files."""
        summary_path = self.workspace_dir / "development_summary.md"
        
        with open(summary_path, "w", encoding="utf-8") as f:  # Add UTF-8 encoding here
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
    _current_playback: threading.Event = None
    _playback_queue: Queue = None
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._current_playback = threading.Event()
        self._current_playback.set()  # No playback in progress initially
        self._playback_queue = Queue()
        # Start the playback worker thread
        self._start_playback_worker()
    
    def _start_playback_worker(self):
        """Start a worker thread to handle TTS playback."""
        def playback_worker():
            while True:
                message, voice, speed = self._playback_queue.get()
                if message is None:  # Poison pill to stop the thread
                    break
                
                self._current_playback.clear()  # Mark playback as in progress
                
                try:
                    # Actually play the audio
                    print(f"\n🔊 {message}")
                    text_to_speech(
                        text=message,
                        voice=voice,
                        speed=speed,
                        auto_play=True
                    )
                finally:
                    self._current_playback.set()  # Mark playback as complete
                    self._playback_queue.task_done()
        
        thread = threading.Thread(target=playback_worker, daemon=True)
        thread.start()
    
    def narrate(self, message: str, force: bool = False, wait_for_previous: bool = True):
        """Convert text to speech and play it."""
        # Skip if this exact message was recently narrated
        if not force and message in self.last_messages:
            return
            
        # Keep track of recent messages (simple deduplication)
        self.last_messages.add(message)
        if len(self.last_messages) > 10:
            self.last_messages.pop()
        
        # If requested, wait for any current playback to finish
        if wait_for_previous:
            self._current_playback.wait()
            
        # Queue the message for playback
        self._playback_queue.put((message, self.voice, self.speed))
    
    def announce_project_start(self, project_name: str):
        """Announce the start of a project."""
        self.narrate(f"Starting development of {project_name}.", force=True, wait_for_previous=True)
    
    def announce_planning(self):
        """Announce that planning is in progress."""
        self.narrate("Generating development plan.", force=True, wait_for_previous=True)
    
    def announce_task_start(self, task: DevTask):
        """Announce the start of a task."""
        self.narrate(f"Starting implementation of {task.name}.", wait_for_previous=True)
    
    def announce_task_completion(self, task: DevTask):
        """Announce the completion of a task."""
        # Less critical, can overlap with other activities
        self.narrate(f"Completed task: {task.name}.", wait_for_previous=False)
    
    def announce_milestone(self, completion_percentage: float):
        """Announce when a milestone is reached."""
        if completion_percentage % 25 == 0:
            self.narrate(f"Milestone reached: {int(completion_percentage)}% of tasks completed.", wait_for_previous=True)
    
    def announce_project_completion(self, project_name: str, workspace_path: str):
        """Announce the completion of the project."""
        self.narrate(f"Project {project_name} has been successfully completed! All code files have been saved to {workspace_path}.", force=True, wait_for_previous=True)


# =====================================================
# LLM INTEGRATION
# =====================================================

class LLMDeveloper(BaseModel):
    """Uses LLM to generate code and evaluate progress."""
    system_prompt: str = "You are an expert Python developer."
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
            
            Break it down into 3-6 tasks with clear names, descriptions, priority order (1 being highest),
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
            Message(role="system", content=f"""You are an expert Python developer specialized in production-ready, Pydantic-based implementations.
                    Your task is to write COMPLETE, FUNCTIONAL Python code that can be run immediately.
                    DO NOT write placeholders or TODO comments - implement FULL working code.
                    Your implementation should be Pythonic, well-structured, and include proper error handling.
                    If imports are needed, include them. If classes are needed, define them completely."""),
            Message(role="user", content=f"""
            Task: {task.name}
            Description: {task.description}

            Existing code:
            {code_context or "No existing code yet."}

            Write a COMPLETE, EXECUTABLE implementation for this task. 
            Include all necessary imports, full class definitions, and detailed implementations.
            DO NOT use placeholder implementations or TODOs - write real, production-quality code.

            Return the implementation in this format:
            <filename>
            ```python
            # Complete code implementation here
            ```
            """)
        ]
        
        # Get streaming response
        response = self.client.chat_completion(
            messages=messages,
            temperature=0.5,
            max_tokens=-1,
            stream=True
        )
        
        # Process streaming response
        full_response = ""
        print(f"\n📝 Generating code for: {task.name}")
        print("=" * 50)
        
        try:
            # Use the streaming process method from the client
            for chunk in self.client.stream_generator(response):
                self.stream_printer.print_stream(chunk, prefix="")
                full_response += chunk
            
            print("\n" + "=" * 50)
            
            # Extract filename and code
            lines = full_response.strip().split("\n")
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
                code_content = f"""
# Implementation for {task.name}

def main():
    print("Implementing {task.name}")
    
    # TODO: Add actual implementation here
    
if __name__ == "__main__":
    main()
"""
            
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
                max_tokens=100,
                stream=False
            )
            
            return summary.strip()
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Implemented {task.name}"
    
    def generate_actions(self, state: DevState, n: int = 3) -> List[DevAction]:
        """Generate possible next actions from a state using the LLM."""
        next_task = state.get_next_task()
        if not next_task:
            return []
            
        # Format existing code context for the LLM prompt
        code_context = "\n\n".join([
            f"File: {filename}\n```python\n{content}\n```" 
            for filename, content in state.code_artifacts.items()
        ])
        
        try:
            # Create a proper implementation for the action via the LLM
            filename = f"{next_task.name.lower().replace(' ', '_')}.py"
            
            # Generate real working code via the LLM
            messages = [
                Message(role="system", content=f"""You are an expert Python developer specializing in production-ready, Pydantic-based implementations.
                        Your task is to write COMPLETE, FUNCTIONAL Python code that can be run immediately.
                        DO NOT write placeholders or TODO comments - implement FULL working code.
                        Your implementation should be Pythonic, well-structured, and include proper error handling.
                        If imports are needed, include them. If classes are needed, define them completely."""),
                Message(role="user", content=f"""
                        Task: {next_task.name}
                        Description: {next_task.description}

                        Existing code:
                        {code_context or "No existing code yet."}

                        Write a COMPLETE, EXECUTABLE implementation for this task. 
                        Include all necessary imports, full class definitions, and detailed implementations.
                        DO NOT use placeholder implementations or TODOs - write real, production-quality code.
                        """)
            ]
            
            # Get content for this action
            content = self.client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=-1,
                stream=False
            )
            
            # Create the main write_code action
            main_action = DevAction(
                action_type=ActionType.WRITE_CODE,
                target=filename,
                content=content
            )
            
            # Also create some refactor actions if we have existing code
            refactor_actions = []
            if state.code_artifacts:
                existing_files = list(state.code_artifacts.keys())
                if existing_files:
                    # Pick an existing file to potentially refactor
                    existing_file = random.choice(existing_files)
                    
                    # Create a refactor prompt
                    refactor_messages = [
                        Message(role="system", content="You are an expert code refactorer."),
                        Message(role="user", content=f"""
                        Refactor this file to work with the new task:
                        
                        New task: {next_task.name}
                        Description: {next_task.description}
                        
                        File to refactor: {existing_file}
                        Current content:
                        ```python
                        {state.code_artifacts[existing_file]}
                        ```
                        
                        Provide an improved version of this file that works with the new task.
                        Return only the refactored code.
                        """)
                    ]
                    
                    # Get refactored content
                    refactored_content = self.client.chat_completion(
                        messages=refactor_messages,
                        temperature=0.6,
                        max_tokens=-1,
                        stream=False
                    )
                    
                    # Add a refactor action
                    refactor_actions.append(DevAction(
                        action_type=ActionType.REFACTOR,
                        target=existing_file,
                        content=refactored_content
                    ))
            
            # Combine and return actions
            actions = [main_action] + refactor_actions
            return actions[:n]  # Return at most n actions
        
        except Exception as e:
            print(f"❌ Error generating actions: {e}")
            # Return a basic fallback action with imports
            filename = f"{next_task.name.lower().replace(' ', '_')}.py"
            return [DevAction(
                action_type=ActionType.WRITE_CODE,
                target=filename,
                content=f"""
            import os
            import json
            from typing import List, Dict, Any, Optional
            from pathlib import Path

            def implement_{next_task.name.lower().replace(' ', '_')}():
                \"\"\"
                Implementation for: {next_task.name}
                {next_task.description}
                \"\"\"
                print(f"Implementing {next_task.name}")
                
                # TODO: Replace with real implementation
                
                return True

            if __name__ == "__main__":
                implement_{next_task.name.lower().replace(' ', '_')}()
            """
            )]
    
    def evaluate_state(self, state: DevState) -> float:
        """Evaluate the quality of the development state using the LLM."""
        # Format code context for evaluation
        code_context = "\n\n".join([
            f"File: {filename}\n```python\n{content}\n```" 
            for filename, content in state.code_artifacts.items()
        ])
        
        # Format requirements for context
        completed_reqs = "\n".join([f"- ✅ {task.name}" for task in state.tasks if task.completed])
        pending_reqs = "\n".join([f"- ⬜ {task.name}" for task in state.tasks if not task.completed])
        
        try:
            # Create evaluation prompt
            messages = [
                Message(role="system", content="You are an expert code quality evaluator. Your job is to analyze code and provide a numeric score from 0.0 to 1.0 based on completeness, correctness, and quality."),
                Message(role="user", content=f"""
                Evaluate the current state of this development project.

                PROJECT REQUIREMENTS:
                Completed:
                {completed_reqs or "None yet"}
                
                Pending:
                {pending_reqs or "All complete!"}

                CURRENT CODE:
                {code_context or "No code artifacts yet."}

                Provide a quality score from 0.0 to 1.0 where:
                - 0.0 means completely inadequate or missing implementation
                - 0.5 means partial implementation with significant gaps
                - 0.7 means mostly complete with minor issues
                - 0.9 means high-quality implementation with minimal issues
                - 1.0 means perfect implementation

                Respond with ONLY a single number between 0.0 and 1.0 representing your evaluation.
                """)
            ]
            
            # Get evaluation from LLM
            evaluation_response = self.client.chat_completion(
                messages=messages,
                temperature=0.3,  # Low temperature for more consistent evaluations
                max_tokens=50,
                stream=False
            )
            
            # Extract numeric score from response
            try:
                # Find the first floating point number in the response
                import re
                matches = re.findall(r"[0-9]*\.?[0-9]+", evaluation_response)
                if matches:
                    score = float(matches[0])
                    # Ensure the score is within valid range
                    return max(0.0, min(1.0, score))
            except Exception as inner_e:
                print(f"⚠️ Error parsing evaluation score: {inner_e}")
            
            # Fallback to a completion-based score if parsing fails
            completion = state.get_completion_percentage() / 100.0
            return completion * 0.8  # Slightly pessimistic fallback
            
        except Exception as e:
            print(f"❌ Error during state evaluation: {e}")
            # Fallback to completion percentage if LLM evaluation fails
            completion = state.get_completion_percentage() / 100.0
            return completion * 0.7  # More pessimistic for error case


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


class MCTSPlanner(BaseModel):
    """MCTS-based planner for development actions."""
    llm_developer: 'LLMDeveloper'
    exploration_weight: float = Field(1.0, description="Exploration weight (UCB1 constant)")
    simulation_depth: int = Field(3, description="Max depth for MCTS simulations")
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def select_action(self, state: DevState, iterations: int = 10) -> Optional[DevAction]:
        """Use MCTS to select the best next action."""
        # Print status
        print("\n⚙️ Using Monte Carlo Tree Search to find the best next action...")
        
        # Check if state is already complete or has no next task
        next_task = state.get_next_task()
        if state.is_complete() or not next_task:
            print("⚠️ No valid actions available - state is complete or has no available tasks")
            return None
            
        # Create root node
        root = MCTSNode(state=state)
        
        for i in range(iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal() and node.visits > 0:
                node = self._expand(node)
            
            # Simulation
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
            
            if i % 5 == 0:  # Print progress periodically
                print(f"  MCTS iteration {i+1}/{iterations} - Current best value: {root.value/max(1, root.visits):.3f}")
        
        # Select the child with the highest value
        if not root.children:
            # If no children, create a basic action for the next task
            print("⚠️ MCTS found no valid actions - generating fallback action")
            task = state.get_next_task()
            if task:
                filename = f"{task.name.lower().replace(' ', '_')}.py"
                content = f"""
                        # Implementation for {task.name}

                        def main():
                            print("Implementing {task.name}")
                            
                            # TODO: Add actual implementation here
                            
                        if __name__ == "__main__":
                            main()
                        """
                return DevAction(
                    action_type=ActionType.WRITE_CODE,
                    target=filename,
                    content=content
                )
            return None
            
        # Use visits count as primary selection criterion
        best_child = max(root.children, key=lambda c: c.visits)
        print(f"✅ MCTS selected action: {best_child.action.action_type} for {best_child.action.target}")
        
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB."""
        current = node
        
        while current.children and not current.is_terminal():
            # If any children are not visited, select one of them
            unvisited = [child for child in current.children if child.visits == 0]
            if unvisited:
                return random.choice(unvisited)
                
            # Otherwise select according to UCB score
            current = max(current.children, key=lambda c: c.ucb_score(self.exploration_weight))
            
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand by adding child nodes for all possible actions."""
        if node.is_terminal() or node.is_fully_expanded():
            return node
            
        # Generate possible actions from this state
        actions = self.llm_developer.generate_actions(node.state)
        
        for action in actions:
            # Apply action to get new state
            new_state = action.apply_to_state(node.state)
            
            # Create child node
            child = MCTSNode(
                state=new_state,
                parent=node,
                action=action
            )
            
            # Add to children
            node.children.append(child)
            
        # Return a random child if any were created
        if node.children:
            return random.choice(node.children)
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a random playout from this node."""
        # If terminal node, return evaluation
        if node.is_terminal():
            return self.llm_developer.evaluate_state(node.state)
            
        # Start with current state
        current_state = node.state.model_copy(deep=True)
        depth = 0
        
        # Simulate until terminal state or max depth
        while not current_state.is_complete() and depth < self.simulation_depth:
            # Get possible actions
            actions = self.llm_developer.generate_actions(current_state)
            if not actions:
                break
                
            # Choose random action
            action = random.choice(actions)
            
            # Apply action
            current_state = action.apply_to_state(current_state)
            
            # Increment depth
            depth += 1
            
        # Evaluate final state
        return self.llm_developer.evaluate_state(current_state)
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate the evaluation up the tree."""
        # Update all nodes up to the root
        current = node
        while current:
            current.visits += 1
            current.value += value
            current = current.parent


class AStarPlanner(BaseModel):
    """A* search-based planner for finding optimal implementation paths."""
    llm_developer: 'LLMDeveloper'
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    
    def plan_implementation(self, initial_state: DevState, max_iterations: int = 50) -> List[DevAction]:
        """Find an optimal sequence of actions to complete implementation."""
        print("\n🧩 Planning optimal implementation path with A* search...")
        
        # Check if state is already complete
        if initial_state.is_complete():
            print("⚠️ State is already complete - no planning needed")
            return []
            
        # If there are no tasks, nothing to plan
        if not initial_state.tasks:
            print("⚠️ No tasks to plan")
            return []
        
        # Create open and closed sets
        open_set = []
        closed_set = set()
        
        # Create start node
        start_node = AStarNode(state=initial_state)
        
        # Add start node to open set (as a tuple for heapq)
        heapq.heappush(open_set, (0, id(start_node), start_node))
        
        # Track nodes by state hash for fast lookup
        state_to_node = {initial_state.get_state_hash(): start_node}
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            _, _, current = heapq.heappop(open_set)
            current_hash = current.state.get_state_hash()
            
            # Print progress
            if iterations % 10 == 0:
                print(f"  A* iteration {iterations} - Nodes explored: {len(closed_set)}, Open nodes: {len(open_set)}")
            
            # Remove from the lookup dict
            if current_hash in state_to_node:
                del state_to_node[current_hash]
            
            # Add to closed set
            closed_set.add(current_hash)
            
            # Check if we've reached the goal
            if current.state.is_complete():
                print(f"✅ A* found optimal path in {iterations} iterations")
                return self._reconstruct_path(current)
            
            # Generate actions for next task
            next_task = current.state.get_next_task()
            if not next_task:
                continue
                
            # Generate actions for this task
            for action in self.llm_developer.generate_actions(current.state):
                # Apply action to get new state
                new_state = action.apply_to_state(current.state)
                new_hash = new_state.get_state_hash()
                
                # Skip if in closed set
                if new_hash in closed_set:
                    continue
                
                # Calculate costs
                g_cost = current.g_cost + self._action_cost(action)
                h_cost = self._heuristic(new_state)
                f_cost = g_cost + h_cost
                
                # Check if we already have this state in the open set
                if new_hash in state_to_node:
                    # If we found a better path, update it
                    existing_node = state_to_node[new_hash]
                    if g_cost < existing_node.g_cost:
                        # Update the existing node
                        existing_node.parent = current
                        existing_node.g_cost = g_cost
                        existing_node.action = action
                        
                        # Re-add to the heap with new priority
                        heapq.heappush(open_set, (f_cost, id(existing_node), existing_node))
                else:
                    # Create new node
                    new_node = AStarNode(
                        state=new_state,
                        parent=current,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        action=action
                    )
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_cost, id(new_node), new_node))
                    state_to_node[new_hash] = new_node
        
        print(f"⚠️ A* could not find a complete solution in {max_iterations} iterations")
        
        # Return partial plan if available
        if open_set:
            best_node = min(open_set, key=lambda x: x[0])[2]
            return self._reconstruct_path(best_node)
            
        # Return empty list if no plan
        return []
    
    def _heuristic(self, state: DevState) -> float:
        """Heuristic function estimating cost to goal."""
        # Basic heuristic: remaining percentage of tasks
        completion = state.get_completion_percentage() / 100.0
        return 1.0 - completion
    
    def _action_cost(self, action: DevAction) -> float:
        """Calculate the cost of an action."""
        # Different actions have different costs
        if action.action_type == ActionType.WRITE_CODE:
            return 1.0
        elif action.action_type == ActionType.REFACTOR:
            return 0.7
        elif action.action_type == ActionType.TEST:
            return 0.3
        elif action.action_type == ActionType.DOCUMENT:
            return 0.5
        else:
            return 1.0
    
    def _reconstruct_path(self, end_node: 'AStarNode') -> List[DevAction]:
        """Reconstruct the path from the goal node to the start."""
        actions = []
        current = end_node
        
        while current.parent and current.action:
            actions.insert(0, current.action)  # Insert at beginning to reverse order
            current = current.parent
            
        return actions


class AStarNode(BaseModel):
    """A* search node for finding optimal implementation paths."""
    state: DevState
    parent: Optional['AStarNode'] = None
    g_cost: float = Field(0.0, description="Cost from start to current node")
    h_cost: float = Field(0.0, description="Heuristic cost to goal")
    action: Optional[DevAction] = None
    
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)


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
    max_iterations: int = Field(7, description="Maximum number of iterations to prevent infinite loops")
    
    # Components (initialized later)
    narrator: Optional[VoiceNarrator] = None
    developer: Optional[LLMDeveloper] = None
    mcts_planner: Optional[MCTSPlanner] = None
    astar_planner: Optional[AStarPlanner] = None
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
            system_prompt="You are an expert Python developer specializing in clean, modular code.",
            stream_printer=self.stream_printer
        )
        
        # Initialize state
        self.state = DevState(project_name=self.project_name)
        
        # Initialize workspace
        self.workspace = WorkspaceManager(project_name=self.project_name)
        
        # Initialize planners (note that these depend on the developer being initialized first)
        self.mcts_planner = MCTSPlanner(llm_developer=self.developer)
        self.astar_planner = AStarPlanner(llm_developer=self.developer)
    
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
                filename, code = self.developer.implement_task(next_task, self.state.current_code)
                
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


def main():
    """Run the agentic development system with search algorithms."""
    project_name = "Smart Task Manager"
    project_description = """
    Create a simple task management system that uses the LLM to prioritize tasks and
    the TTS system to provide verbal notifications about important tasks. The system
    should:
    1. Allow adding tasks with descriptions
    2. Use the LLM to categorize and prioritize tasks
    3. Use TTS to announce high-priority tasks
    4. Provide a simple command-line interface
    """
    
    # Initialize the system
    system = AgenticDevelopmentSystem(
        project_name=project_name,
        project_description=project_description,
        voice="af_bella",
        voice_speed=0.9,
        typing_speed=0.001,  # Adjust typing speed (lower = faster)
        use_astar_planning=True,
        use_mcts_actions=True,
        max_iterations=7  # Limit the number of iterations to prevent infinite loops
    )
    
    # Execute development process
    code_artifacts = system.develop()
    
    # Print the workspace location
    workspace_path = system.workspace.get_workspace_path()
    print("\n" + "="*50)
    print(f"📁 All code generated in: {workspace_path}")
    print("="*50)


if __name__ == "__main__":
    main()