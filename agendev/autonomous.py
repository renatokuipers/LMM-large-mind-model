"""Autonomous development process functionality for AgenDev."""

import logging
import threading
import time
from datetime import datetime
from uuid import UUID
import os
from pathlib import Path

from agendev.models.task_models import TaskType, TaskStatus, TaskPriority, TaskRisk
from agendev.llm_module import Message
# Import advanced modules
from agendev.context_management import ContextManager
from agendev.probability_modeling import TaskProbabilityModel, ProjectRiskModel
from agendev.search_algorithms import MCTSPlanner, AStarPathfinder
from agendev.parameter_controller import ParameterController
from agendev.snapshot_engine import SnapshotEngine
from agendev.test_generation import TestGenerator, TestType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for autonomous process
autonomous_thread = None
autonomous_running = False
user_responses = {}
user_questions = []

def generate_tasks_from_description(agendev, project_name, project_description):
    """Generate epics and tasks based on project description using LLM."""
    logger.info(f"Generating tasks for project: {project_name}")
    
    # Use the LLM to analyze the project description and generate tasks
    try:
        # Create a system message for the LLM
        system_message = "You are an expert software architect and project planner. Your task is to analyze a project description and create a structured plan with epics and tasks."
        
        # Create prompt for the LLM
        prompt = f"""
        Project Name: {project_name}
        Project Description: {project_description}
        
        Based on this description, please create a comprehensive plan with:
        1. 3-10 high-level epics that represent major components or milestones
        2. 3-10 specific tasks for each epic
        
        For each epic, provide:
        - Title
        - Description
        - Priority (HIGH, MEDIUM, or LOW)
        - Risk level (HIGH, MEDIUM, or LOW)
        
        For each task, provide:
        - Title
        - Description
        - Type (IMPLEMENTATION, REFACTOR, BUGFIX, TEST, DOCUMENTATION, or PLANNING)
        - Priority (HIGH, MEDIUM, or LOW)
        - Risk level (HIGH, MEDIUM, or LOW)
        - Estimated duration in hours
        - Dependencies (if any, by task title)
        
        Structure your response as a JSON object.
        """
        
        # Define schema for structured output
        json_schema = {
            "name": "project_plan",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "epics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "risk": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                "tasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "type": {"type": "string", "enum": ["IMPLEMENTATION", "REFACTOR", "BUGFIX", "TEST", "DOCUMENTATION", "PLANNING"]},
                                            "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                            "risk": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                            "estimated_duration_hours": {"type": "number"},
                                            "dependencies": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["title", "description", "type", "priority", "risk", "estimated_duration_hours"]
                                    }
                                }
                            },
                            "required": ["title", "description", "priority", "risk", "tasks"]
                        }
                    }
                },
                "required": ["epics"]
            }
        }
        
        # Make the request to the LLM
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=prompt)
        ]
        
        # Get structured response
        response = agendev.llm.llm_client.structured_completion(
            messages=messages,
            json_schema=json_schema,
            temperature=0.7
        )
        
        # Check if response is valid
        if not response or not isinstance(response, dict):
            logger.error(f"Invalid response from LLM: {response}")
            if agendev.notification_manager:
                agendev.notification_manager.error("Failed to generate tasks: Invalid response from LLM")
            return {"success": False, "error": "Invalid response from LLM"}
        
        # Parse the response - handle both dict and potentially parsed JSON string
        if "project_plan" in response:
            # Extract from wrapped project_plan if present
            response_data = response["project_plan"]
        else:
            # Use the response directly
            response_data = response
            
        # Check if the expected structure exists
        if not isinstance(response_data, dict) or "epics" not in response_data:
            logger.error(f"Invalid response structure from LLM: {response_data}")
            if agendev.notification_manager:
                agendev.notification_manager.error("Failed to generate tasks: Invalid response structure")
            return {"success": False, "error": "Invalid response structure"}
        
        # Process the response to create epics and tasks
        epic_ids = []
        task_ids = []
        task_title_to_id = {}
        
        # Create epics
        for epic_data in response_data.get("epics", []):
            epic_id = agendev.create_epic(
                title=epic_data["title"],
                description=epic_data["description"],
                priority=TaskPriority(epic_data["priority"].lower()),
                risk=TaskRisk(epic_data["risk"].lower())
            )
            epic_ids.append(epic_id)
            
            # Create tasks for this epic
            for task_data in epic_data.get("tasks", []):
                task_id = agendev.create_task(
                    title=task_data["title"],
                    description=task_data["description"],
                    task_type=TaskType(task_data["type"].lower()),
                    priority=TaskPriority(task_data["priority"].lower()),
                    risk=TaskRisk(task_data["risk"].lower()),
                    estimated_duration_hours=task_data["estimated_duration_hours"],
                    epic_id=epic_id
                )
                task_ids.append(task_id)
                task_title_to_id[task_data["title"]] = task_id
        
        # Process dependencies (second pass)
        for epic_data in response_data.get("epics", []):
            for task_data in epic_data.get("tasks", []):
                if "dependencies" in task_data and task_data["dependencies"]:
                    task_id = task_title_to_id.get(task_data["title"])
                    if task_id:
                        for dep_title in task_data["dependencies"]:
                            dep_id = task_title_to_id.get(dep_title)
                            if dep_id and dep_id in agendev.task_graph.tasks:
                                # Add dependency relationship
                                agendev.task_graph.tasks[task_id].dependencies.append(dep_id)
                                agendev.task_graph.tasks[dep_id].dependents.append(task_id)
        
        # Update task statuses based on dependencies
        agendev.task_graph.update_task_statuses()
        
        # Notify about plan generation
        if agendev.notification_manager:
            agendev.notification_manager.success(
                f"Generated project plan with {len(epic_ids)} epics and {len(task_ids)} tasks."
            )
            
        return {
            "success": True,
            "epic_count": len(epic_ids),
            "task_count": len(task_ids),
            "epic_ids": [str(eid) for eid in epic_ids],
            "task_ids": [str(tid) for tid in task_ids]
        }
    
    except Exception as e:
        logger.error(f"Error generating tasks from description: {e}")
        if agendev.notification_manager:
            agendev.notification_manager.error(f"Failed to generate tasks: {e}")
        return {"success": False, "error": str(e)}

def autonomous_development_process(agendev):
    """Main function for autonomous development process."""
    global autonomous_running, user_questions, user_responses
    
    try:
        # Notify start
        if agendev.notification_manager:
            agendev.notification_manager.info("Starting autonomous development process...")
        
        # ENHANCEMENT: Use context management to build project understanding
        build_project_context(agendev)
        
        # Step 1: Generate implementation plan
        logger.info("Generating implementation plan...")
        plan = agendev.generate_implementation_plan(max_iterations=500)
        
        if not plan:
            raise Exception("Failed to generate implementation plan")
            
        # Notify about plan generation
        if agendev.notification_manager:
            agendev.notification_manager.success(
                f"Implementation plan generated with {len(plan.task_sequence)} tasks."
            )
        
        # ENHANCEMENT: Analyze project risk using probability modeling
        risk_report = analyze_project_risk(agendev, plan)
        logger.info(f"Project risk analysis: Success probability {risk_report['success_probability']:.2f}")
        
        # Step 2: Implement tasks in sequence
        for task_id in plan.task_sequence:
            # Check if we need to ask the user any questions before implementing this task
            task = agendev.task_graph.tasks.get(task_id)
            if not task:
                continue
                
            logger.info(f"Processing task: {task.title}")
            
            # ENHANCEMENT: Use parameter controller to optimize LLM settings for task
            optimize_llm_parameters(agendev, task)
            
            # Check if this is a complex task that might need clarification
            if task.risk in [TaskRisk.HIGH, TaskRisk.CRITICAL] or task.priority == TaskPriority.CRITICAL:
                # Generate a question for the user
                question = generate_question_for_task(agendev, task)
                if question:
                    # Add to questions queue
                    user_questions.append({
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "question": question,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Notify about question
                    if agendev.notification_manager:
                        agendev.notification_manager.info(
                            f"Question about task '{task.title}'. Please check the dashboard."
                        )
                    
                    # Wait for response
                    wait_time = 0
                    max_wait = 300  # 5 minutes max wait
                    while str(task_id) not in user_responses and wait_time < max_wait:
                        time.sleep(5)
                        wait_time += 5
                        if not autonomous_running:
                            return  # Exit if process was stopped
            
            # Implement the task
            try:
                # Check if we have user input to consider
                additional_context = user_responses.get(str(task_id), "")
                
                # Implement with additional context if available
                result = implement_task_with_context(agendev, task_id, additional_context)
                
                # ENHANCEMENT: Take code snapshot after implementation
                create_implementation_snapshot(agendev, task_id, result)
                
                # ENHANCEMENT: Generate tests for implemented code
                generate_tests_for_task(agendev, task_id)
                
                # Notify about implementation
                if agendev.notification_manager:
                    agendev.notification_manager.success(
                        f"Task '{task.title}' implemented successfully."
                    )
            except Exception as e:
                logger.error(f"Error implementing task {task.title}: {e}")
                if agendev.notification_manager:
                    agendev.notification_manager.error(f"Failed to implement task '{task.title}': {e}")
        
        # Notify about completion
        if agendev.notification_manager:
            agendev.notification_manager.milestone("Project implementation completed!")
            
        # Generate final summary
        summary = agendev.summarize_progress(voice_summary=True)
        
    except Exception as e:
        logger.error(f"Error in autonomous development process: {e}")
        if agendev.notification_manager:
            agendev.notification_manager.error(f"Autonomous development process failed: {e}")
    finally:
        autonomous_running = False

def build_project_context(agendev):
    """Build code context for better understanding of the project."""
    logger.info("Building project context using context management...")
    
    try:
        # Get project workspace directory
        workspace_dir = agendev.workspace_dir / "src"
        if not workspace_dir.exists():
            workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Use context manager to index relevant files
        index_count = 0
        for root, _, files in os.walk(workspace_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        elements = agendev.context_manager.index_file(
                            file_path, 
                            element_types=["file", "class", "function", "method"]
                        )
                        index_count += len(elements)
                    except Exception as e:
                        logger.warning(f"Error indexing file {file_path}: {e}")
        
        logger.info(f"Indexed {index_count} code elements for context")
        return {"success": True, "indexed_elements": index_count}
    
    except Exception as e:
        logger.error(f"Error building project context: {e}")
        return {"success": False, "error": str(e)}

def analyze_project_risk(agendev, plan):
    """Analyze project risk using probability modeling."""
    logger.info("Analyzing project risk...")
    
    try:
        # Create task probability model
        probability_model = agendev.probability_model if hasattr(agendev, 'probability_model') else None
        
        if not probability_model:
            probability_model = TaskProbabilityModel(
                task_graph=agendev.task_graph,
                llm_integration=agendev.llm
            )
        
        # Create project risk model
        risk_model = ProjectRiskModel(
            task_probability_model=probability_model,
            task_graph=agendev.task_graph
        )
        
        # Calculate overall project success probability
        success_probability = risk_model.calculate_project_success_probability()
        
        # Identify risk hotspots
        risk_hotspots = risk_model.identify_risk_hotspots(threshold=0.7)
        
        # Run Monte Carlo simulation
        simulation_results = risk_model.monte_carlo_simulation(num_simulations=100)
        
        # Create risk report
        report = {
            "success_probability": success_probability,
            "risk_hotspots": risk_hotspots,
            "simulation_results": {
                "mean_completion_time": simulation_results["mean_completion_time"],
                "completion_probability": simulation_results["completion_probability"]
            }
        }
        
        return report
    
    except Exception as e:
        logger.error(f"Error analyzing project risk: {e}")
        return {"success_probability": 0.5, "error": str(e)}

def optimize_llm_parameters(agendev, task):
    """Optimize LLM parameters for specific task using parameter controller."""
    logger.info(f"Optimizing LLM parameters for task: {task.title}")
    
    try:
        # Get optimized LLM configuration for this task
        llm_config = agendev.parameter_controller.get_llm_config(task)
        
        # Apply configuration to LLM integration
        # This doesn't change the actual LLM parameters but prepares them for the next call
        task_specific_config = agendev.parameter_controller.get_profile_for_task(task)
        
        logger.info(f"Optimized parameters for {task.task_type.value} task: "
                   f"temperature={task_specific_config.temperature}, "
                   f"max_tokens={task_specific_config.max_tokens}")
        
        return {"success": True, "config": llm_config}
    
    except Exception as e:
        logger.error(f"Error optimizing LLM parameters: {e}")
        return {"success": False, "error": str(e)}

def create_implementation_snapshot(agendev, task_id, implementation_result):
    """Create a snapshot of code after implementing a task."""
    task = agendev.task_graph.tasks.get(task_id)
    if not task:
        return {"success": False, "error": "Task not found"}
    
    try:
        # Get artifact paths
        if not task.artifact_paths:
            return {"success": False, "error": "No artifacts found for task"}
        
        snapshots = []
        
        # Create a snapshot for each artifact
        for path in task.artifact_paths:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                
                # Create snapshot
                snapshot = agendev.snapshot_engine.create_snapshot(
                    file_path=path,
                    content=content,
                    commit_message=f"Implementation of task: {task.title}",
                    tags=[task.task_type.value, f"priority-{task.priority.value}", f"task-{task_id}"]
                )
                
                snapshots.append(snapshot.snapshot_id)
                logger.info(f"Created snapshot {snapshot.snapshot_id} for file {path}")
            
            except Exception as e:
                logger.warning(f"Error creating snapshot for {path}: {e}")
        
        return {"success": True, "snapshots": snapshots}
    
    except Exception as e:
        logger.error(f"Error creating implementation snapshot: {e}")
        return {"success": False, "error": str(e)}

def generate_tests_for_task(agendev, task_id):
    """Generate automated tests for implemented task."""
    task = agendev.task_graph.tasks.get(task_id)
    if not task or not task.artifact_paths:
        return {"success": False, "error": "Task not found or has no artifacts"}
    
    try:
        test_files = []
        
        # Generate tests for each artifact
        for path in task.artifact_paths:
            if not path.endswith('.py'):
                continue
                
            try:
                # Extract code elements from the file
                elements = agendev.test_generator.extract_code_elements(path)
                
                if not elements:
                    logger.warning(f"No code elements found in {path}")
                    continue
                
                # Generate test suite
                test_suite = agendev.test_generator.generate_test_suite(
                    path,
                    test_types=[TestType.UNIT, TestType.INTEGRATION]
                )
                
                # Save test suite
                test_file_path = agendev.test_generator.save_test_suite(test_suite)
                test_files.append(test_file_path)
                
                logger.info(f"Generated test suite {test_file_path} for {path}")
                
            except Exception as e:
                logger.warning(f"Error generating tests for {path}: {e}")
        
        return {"success": True, "test_files": test_files}
    
    except Exception as e:
        logger.error(f"Error generating tests for task: {e}")
        return {"success": False, "error": str(e)}

def generate_question_for_task(agendev, task):
    """Generate a question to ask the user about a complex task."""
    try:
        # Use LLM to generate a relevant question
        system_message = "You are an AI assistant helping with software development. Generate a specific question to ask the user about this task."
        
        prompt = f"""
        Task: {task.title}
        Description: {task.description}
        Priority: {task.priority.value}
        Risk: {task.risk.value}
        
        This task has been identified as complex or critical. Please generate a specific question to ask the user 
        that would help clarify requirements or provide guidance for implementing this task.
        
        The question should be focused, specific, and directly related to implementing this particular task.
        """
        
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=prompt)
        ]
        
        question = agendev.llm.llm_client.chat_completion(messages=messages, temperature=0.7)
        return question
    
    except Exception as e:
        logger.error(f"Error generating question for task: {e}")
        return None

def implement_task_with_context(agendev, task_id, additional_context=""):
    """Implement a task with additional context from user."""
    if task_id not in agendev.task_graph.tasks:
        raise Exception(f"Task not found: {task_id}")
    
    task = agendev.task_graph.tasks[task_id]
    
    # Create enhanced implementation with additional context
    if additional_context:
        # Add the additional context to the task description
        enhanced_task = task.model_copy()
        enhanced_task.description = f"{task.description}\n\nAdditional context from user: {additional_context}"
        
        # Store the original task
        original_task = agendev.task_graph.tasks[task_id]
        
        # Temporarily replace with enhanced task
        agendev.task_graph.tasks[task_id] = enhanced_task
        
        # Implement the task
        result = agendev.implement_task(task_id)
        
        # Restore original task (but keep the status and artifacts)
        original_task.status = enhanced_task.status
        original_task.completion_percentage = enhanced_task.completion_percentage
        original_task.actual_duration_hours = enhanced_task.actual_duration_hours
        original_task.artifact_paths = enhanced_task.artifact_paths
        agendev.task_graph.tasks[task_id] = original_task
    else:
        # Implement normally
        result = agendev.implement_task(task_id)
    
    return result

def start_autonomous_process(agendev):
    """Start the autonomous development process in a separate thread."""
    global autonomous_thread, autonomous_running
    
    if autonomous_running:
        return {"success": False, "message": "Autonomous process already running"}
    
    autonomous_running = True
    autonomous_thread = threading.Thread(target=autonomous_development_process, args=(agendev,))
    autonomous_thread.daemon = True
    autonomous_thread.start()
    
    return {"success": True, "message": "Autonomous process started"}

def stop_autonomous_process():
    """Stop the autonomous development process."""
    global autonomous_running
    autonomous_running = False
    return {"success": True, "message": "Autonomous process stopping"}

def get_autonomous_status():
    """Get the current status of the autonomous process."""
    global autonomous_running
    return autonomous_running

def get_user_questions():
    """Get the list of user questions."""
    global user_questions
    return user_questions

def add_user_response(task_id, response):
    """Add a user response to a task question."""
    global user_responses
    user_responses[task_id] = response
    return True

def generate_alternative_plan(agendev, optimization_goal='speed'):
    """Generate an alternative implementation plan using A* algorithm.
    
    This demonstrates using the AStarPathfinder from search_algorithms.py
    
    Args:
        agendev: AgenDev instance
        optimization_goal: What to optimize for ('speed', 'quality', 'risk')
        
    Returns:
        Alternative plan and comparison with MCTS plan
    """
    logger.info(f"Generating alternative plan optimized for {optimization_goal}...")
    
    try:
        # Create A* pathfinder
        pathfinder = AStarPathfinder(
            task_graph=agendev.task_graph
        )
        
        # Define heuristic based on optimization goal
        if optimization_goal == 'speed':
            def custom_heuristic(node_id, remaining_tasks):
                return sum(agendev.task_graph.tasks[task_id].estimated_duration_hours 
                          for task_id in remaining_tasks)
                          
        elif optimization_goal == 'quality':
            def custom_heuristic(node_id, remaining_tasks):
                # Higher priority tasks should be done first (lower score)
                priority_score = sum(
                    {'low': 3, 'medium': 2, 'high': 1, 'critical': 0}[agendev.task_graph.tasks[task_id].priority.value]
                    for task_id in remaining_tasks
                )
                return priority_score
                
        elif optimization_goal == 'risk':
            def custom_heuristic(node_id, remaining_tasks):
                # Lower risk tasks should be done first (lower score)
                risk_score = sum(
                    {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[agendev.task_graph.tasks[task_id].risk.value]
                    for task_id in remaining_tasks
                )
                return risk_score
        else:
            # Default to critical path heuristic
            custom_heuristic = pathfinder.critical_path_heuristic
        
        # Set custom heuristic
        pathfinder._calculate_heuristic = custom_heuristic
        
        # Find path using A*
        task_sequence, stats = pathfinder.find_path()
        
        # Get current MCTS plan for comparison
        current_plan = agendev.planning_history.get_latest_plan()
        current_sequence = current_plan.task_sequence if current_plan else []
        
        # Compare plans
        if current_sequence:
            # Calculate total duration for both plans
            astar_duration = sum(agendev.task_graph.tasks[task_id].estimated_duration_hours 
                                for task_id in task_sequence)
            mcts_duration = sum(agendev.task_graph.tasks[task_id].estimated_duration_hours 
                               for task_id in current_sequence)
            
            comparison = {
                "astar_task_count": len(task_sequence),
                "mcts_task_count": len(current_sequence),
                "astar_duration": astar_duration,
                "mcts_duration": mcts_duration,
                "duration_diff_percent": (astar_duration - mcts_duration) / mcts_duration * 100
            }
        else:
            comparison = {"astar_task_count": len(task_sequence), "comparison": "No MCTS plan available"}
        
        return {
            "success": True,
            "optimization_goal": optimization_goal,
            "task_sequence": [str(tid) for tid in task_sequence],
            "stats": stats,
            "comparison": comparison
        }
    
    except Exception as e:
        logger.error(f"Error generating alternative plan: {e}")
        return {"success": False, "error": str(e)} 