from typing import Dict, List, Optional, Set, Tuple, Any
import json
import logging
from collections import defaultdict
from datetime import datetime

from planning.models import (
    ProjectPlan,
    Epic,
    Task,
    Subtask,
    PlanReview,
    ComponentType,
    DependencyType
)
from core.llm_manager import LLMManager
from core.exceptions import PlanningError, ParseError

logger = logging.getLogger(__name__)


class TaskReviewer:
    """Reviews and improves project plans.
    
    This class uses an LLM to review project plans for quality, completeness,
    and coherence, then provides suggestions for improvement.
    """
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the task reviewer.
        
        Args:
            llm_manager: LLM manager for reviewing plans
        """
        self.llm_manager = llm_manager
    
    async def review_plan(self, plan: ProjectPlan) -> PlanReview:
        """Review a project plan and identify potential issues.
        
        Args:
            plan: Project plan to review
            
        Returns:
            Plan review results
            
        Raises:
            PlanningError: If review fails
        """
        logger.info(f"Reviewing project plan: {plan.project_name}")
        
        # First, perform automated validation checks
        validation_results = self._perform_automated_validation(plan)
        
        # Then, use the LLM for deeper analysis
        try:
            llm_review = await self._llm_review_plan(plan, validation_results)
            return llm_review
        except Exception as e:
            logger.error(f"Error reviewing plan with LLM: {str(e)}")
            
            # Fall back to just the automated validation results
            return PlanReview(
                plan_id=plan.id,
                issues=validation_results["issues"],
                dependency_issues=validation_results["dependency_issues"],
                complexity_issues=validation_results["complexity_issues"],
                coverage_issues=validation_results["coverage_issues"],
                overall_assessment="Review failed, showing only automated validation results."
            )
    
    def _perform_automated_validation(self, plan: ProjectPlan) -> Dict[str, List[str]]:
        """Perform automated validation checks on the plan.
        
        Args:
            plan: Project plan to validate
            
        Returns:
            Dictionary of validation results by category
        """
        results = {
            "issues": [],
            "dependency_issues": [],
            "complexity_issues": [],
            "coverage_issues": [],
        }
        
        # Check for empty or very sparse epics
        for epic in plan.epics:
            if not epic.tasks:
                results["issues"].append(f"Epic '{epic.title}' has no tasks")
            elif len(epic.tasks) < 2:
                results["issues"].append(f"Epic '{epic.title}' has only {len(epic.tasks)} task(s), which seems too few")
        
        # Check for tasks without requirements
        for task in plan.all_tasks:
            if not task.requirements:
                results["issues"].append(f"Task '{task.title}' has no requirements specified")
        
        # Check for module path consistency
        module_path_counts = defaultdict(int)
        for task in plan.all_tasks:
            base_module = task.module_path.split('.')[0] if '.' in task.module_path else task.module_path
            module_path_counts[base_module] += 1
        
        if len(module_path_counts) == 1:
            results["issues"].append("All tasks use the same base module, which seems unusual for a complex project")
        
        # Validate component types distribution
        component_type_counts = defaultdict(int)
        for task in plan.all_tasks:
            component_type_counts[task.component_type] += 1
        
        # Check for missing essential component types
        essential_types = {
            ComponentType.DATA_MODEL,
            ComponentType.SERVICE,
            ComponentType.API_ENDPOINT
        }
        
        missing_types = essential_types - set(component_type_counts.keys())
        if missing_types:
            formatted_types = ", ".join(str(t.value) for t in missing_types)
            results["coverage_issues"].append(f"Missing essential component types: {formatted_types}")
        
        # Check for imbalanced component types
        if component_type_counts and max(component_type_counts.values()) / sum(component_type_counts.values()) > 0.5:
            dominant_type = max(component_type_counts.items(), key=lambda x: x[1])[0]
            results["coverage_issues"].append(f"Component type '{dominant_type}' dominates the plan, suggesting imbalance")
        
        # Check dependency graph
        dependency_errors = plan.validate_dependencies()
        if dependency_errors:
            results["dependency_issues"].extend(dependency_errors)
        
        # Check for tasks with too many or too few dependencies
        task_map = plan.task_by_id
        for task in plan.all_tasks:
            if len(task.dependencies) > 5:
                results["dependency_issues"].append(f"Task '{task.title}' has {len(task.dependencies)} dependencies, which seems excessive")
                
            # Check for isolated tasks (no dependencies and not depended on)
            if not task.dependencies and not task.dependents:
                results["dependency_issues"].append(f"Task '{task.title}' is isolated (no dependencies or dependents)")
        
        # Check epic ordering and dependencies
        epic_tasks = {epic.id: {task.id for task in epic.tasks} for epic in plan.epics}
        for i, epic in enumerate(plan.epics):
            if i > 0:
                # Check if tasks in this epic depend on tasks from later epics
                prev_epic_tasks = set()
                for prev_epic in plan.epics[:i]:
                    prev_epic_tasks.update(epic_tasks[prev_epic.id])
                    
                for task in epic.tasks:
                    later_dependencies = set(task.dependencies.keys()) - prev_epic_tasks - epic_tasks[epic.id]
                    if later_dependencies:
                        later_dep_tasks = [task_map[dep_id].title for dep_id in later_dependencies if dep_id in task_map]
                        results["dependency_issues"].append(
                            f"Task '{task.title}' in epic {i+1} depends on tasks from later epics: {', '.join(later_dep_tasks)}"
                        )
        
        return results
    
    async def _llm_review_plan(self, 
                              plan: ProjectPlan, 
                              validation_results: Dict[str, List[str]]) -> PlanReview:
        """Use the LLM to review the project plan.
        
        Args:
            plan: Project plan to review
            validation_results: Results from automated validation
            
        Returns:
            Plan review with LLM insights
            
        Raises:
            ParseError: If LLM output cannot be parsed
        """
        # Define the expected response schema for structured output
        review_schema = {
            "type": "object",
            "properties": {
                "issues": {"type": "array", "items": {"type": "string"}},
                "suggestions": {"type": "array", "items": {"type": "string"}},
                "missing_components": {"type": "array", "items": {"type": "string"}},
                "dependency_issues": {"type": "array", "items": {"type": "string"}},
                "complexity_issues": {"type": "array", "items": {"type": "string"}},
                "coverage_issues": {"type": "array", "items": {"type": "string"}},
                "is_approved": {"type": "boolean"},
                "overall_assessment": {"type": "string"}
            },
            "required": ["issues", "suggestions", "overall_assessment"]
        }
        
        # Prepare plan summary for the LLM
        plan_summary = self._create_plan_summary(plan)
        
        # Prepare validation issues
        validation_issues = ""
        for category, issues in validation_results.items():
            if issues:
                validation_issues += f"\n## Automated {category.replace('_', ' ').title()}\n"
                for issue in issues:
                    validation_issues += f"- {issue}\n"
        
        # Build the prompt for review
        prompt = f"""
        # Project Plan Review Task
        
        I need to review a project plan for quality, completeness, and coherence.
        
        ## Project Plan Summary
        
        {plan_summary}
        
        ## Automated Validation Results
        
        The following issues were identified by automated validation:
        {validation_issues if validation_issues else "No issues identified by automated validation."}
        
        ## Review Guidelines
        
        Analyze this project plan critically and provide:
        
        1. Issues: Problems with the current plan that should be addressed
        2. Suggestions: Specific improvements to make the plan better
        3. Missing Components: Important components or functionality that may be missing
        4. Dependency Issues: Problems with task dependencies or sequencing
        5. Complexity Issues: Concerns about complexity estimates or distributions
        6. Coverage Issues: Areas or aspects of the project not adequately covered
        7. Approval Status: Whether the plan is ready for implementation (true/false)
        8. Overall Assessment: A summary of the plan's quality and readiness
        
        Be thorough and specific in your review, focusing on:
        - Completeness: Does the plan cover all aspects of the project?
        - Coherence: Do the epics and tasks form a logical whole?
        - Dependencies: Are task dependencies complete and sensible?
        - Sizing: Are tasks appropriately sized and complexity estimates reasonable?
        - Specificity: Are task descriptions and requirements specific enough?
        - Technical Approach: Is the technical approach sound?
        
        Be specific in your issues and suggestions, referring to specific epics and tasks by name.
        
        ## Expected Output Format
        Return a JSON object with these fields:
        - issues: Array of specific issues with the plan
        - suggestions: Array of specific suggestions for improvement
        - missing_components: Array of components that might be missing
        - dependency_issues: Array of issues with task dependencies
        - complexity_issues: Array of issues with complexity estimates
        - coverage_issues: Array of areas not covered by the plan
        - is_approved: Boolean indicating whether the plan is approved
        - overall_assessment: Overall assessment of the plan
        """
        
        try:
            # Generate review with structured schema
            response = await self.llm_manager.generate(
                prompt=prompt,
                task_type="planning_review",
                schema=review_schema
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                raise ParseError(f"Failed to parse LLM response as JSON: {response.content}")
            
            # Create the review object
            review = PlanReview(
                plan_id=plan.id,
                issues=response_data.get("issues", []),
                suggestions=response_data.get("suggestions", []),
                missing_components=response_data.get("missing_components", []),
                dependency_issues=response_data.get("dependency_issues", []) + validation_results.get("dependency_issues", []),
                complexity_issues=response_data.get("complexity_issues", []) + validation_results.get("complexity_issues", []),
                coverage_issues=response_data.get("coverage_issues", []) + validation_results.get("coverage_issues", []),
                is_approved=response_data.get("is_approved", False),
                overall_assessment=response_data.get("overall_assessment", "No overall assessment provided.")
            )
            
            return review
            
        except Exception as e:
            logger.error(f"Error reviewing plan with LLM: {str(e)}")
            raise
    
    def _create_plan_summary(self, plan: ProjectPlan) -> str:
        """Create a summary of the project plan for the LLM.
        
        Args:
            plan: Project plan to summarize
            
        Returns:
            Formatted plan summary
        """
        summary = f"""
        # Project: {plan.project_name}
        
        Description: {plan.project_description}
        
        ## Epics ({len(plan.epics)})
        """
        
        for i, epic in enumerate(plan.epics):
            summary += f"""
            ### Epic {i+1}: {epic.title}
            
            Description: {epic.description}
            
            Order: {epic.order} | Complexity: {epic.estimated_complexity} | Priority: {epic.priority}
            
            Tasks ({len(epic.tasks)}):
            """
            
            for j, task in enumerate(epic.tasks):
                summary += f"""
                #### Task {j+1}: {task.title}
                
                Description: {task.description}
                
                Type: {task.component_type.value} | Module: {task.module_path} | Name: {task.class_or_function_name or 'Not specified'}
                
                Complexity: {task.estimated_complexity} | Priority: {task.priority}
                
                Requirements:
                {self._format_list(task.requirements)}
                
                Dependencies:
                {self._format_dependencies(task, plan)}
                
                Subtasks:
                {self._format_subtasks(task.subtasks)}
                """
        
        return summary
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a bulleted list.
        
        Args:
            items: List of items
            
        Returns:
            Formatted string
        """
        if not items:
            return "None"
        
        return "\n".join(f"- {item}" for item in items)
    
    def _format_dependencies(self, task: Task, plan: ProjectPlan) -> str:
        """Format task dependencies.
        
        Args:
            task: Task with dependencies
            plan: Project plan
            
        Returns:
            Formatted string
        """
        if not task.dependencies:
            return "None"
        
        task_map = plan.task_by_id
        deps = []
        
        for dep_id, dep_type in task.dependencies.items():
            if dep_id in task_map:
                dep_task = task_map[dep_id]
                deps.append(f"- {dep_task.title} ({dep_type.value})")
            else:
                deps.append(f"- Unknown task ID: {dep_id} ({dep_type.value})")
        
        return "\n".join(deps)
    
    def _format_subtasks(self, subtasks: List[Subtask]) -> str:
        """Format subtasks.
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            Formatted string
        """
        if not subtasks:
            return "None"
        
        return "\n".join(f"- {subtask.title}: {subtask.description}" for subtask in subtasks)
    
    async def improve_plan(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Improve a project plan based on review feedback.
        
        Args:
            plan: Original project plan
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        logger.info(f"Improving project plan based on review: {plan.project_name}")
        
        # If plan is already approved, no need to improve
        if review.is_approved:
            logger.info("Plan is already approved, no improvements needed")
            return plan
        
        # Clone the plan for modification
        improved_plan = ProjectPlan.parse_obj(plan.dict())
        improved_plan.version += 1
        improved_plan.updated_at = datetime.now()
        
        try:
            # For each major issue category, attempt to apply improvements
            if review.dependency_issues:
                improved_plan = await self._improve_dependencies(improved_plan, review)
            
            if review.missing_components:
                improved_plan = await self._add_missing_components(improved_plan, review)
            
            if review.complexity_issues:
                improved_plan = await self._adjust_complexity(improved_plan, review)
            
            if review.coverage_issues:
                improved_plan = await self._improve_coverage(improved_plan, review)
            
            # Apply general improvements based on suggestions
            if review.suggestions:
                improved_plan = await self._apply_general_improvements(improved_plan, review)
            
            return improved_plan
            
        except Exception as e:
            logger.error(f"Error improving plan: {str(e)}")
            raise PlanningError(f"Failed to improve plan: {str(e)}")
    
    async def _improve_dependencies(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Improve task dependencies based on review feedback.
        
        Args:
            plan: Project plan to improve
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        # In a full implementation, this would use the LLM to analyze and fix dependencies
        # For now, we'll just log the issues
        for issue in review.dependency_issues:
            logger.info(f"Dependency issue to fix: {issue}")
        
        return plan
    
    async def _add_missing_components(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Add missing components based on review feedback.
        
        Args:
            plan: Project plan to improve
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        # In a full implementation, this would use the LLM to generate missing components
        # For now, we'll just log the missing components
        for component in review.missing_components:
            logger.info(f"Missing component to add: {component}")
        
        return plan
    
    async def _adjust_complexity(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Adjust complexity estimates based on review feedback.
        
        Args:
            plan: Project plan to improve
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        # In a full implementation, this would use the LLM to suggest complexity adjustments
        # For now, we'll just log the issues
        for issue in review.complexity_issues:
            logger.info(f"Complexity issue to fix: {issue}")
        
        return plan
    
    async def _improve_coverage(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Improve coverage based on review feedback.
        
        Args:
            plan: Project plan to improve
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        # In a full implementation, this would use the LLM to suggest coverage improvements
        # For now, we'll just log the issues
        for issue in review.coverage_issues:
            logger.info(f"Coverage issue to fix: {issue}")
        
        return plan
    
    async def _apply_general_improvements(self, plan: ProjectPlan, review: PlanReview) -> ProjectPlan:
        """Apply general improvements based on review suggestions.
        
        Args:
            plan: Project plan to improve
            review: Plan review with suggestions
            
        Returns:
            Improved project plan
            
        Raises:
            PlanningError: If improvement fails
        """
        # In a full implementation, this would use the LLM to apply suggested improvements
        # For now, we'll just log the suggestions
        for suggestion in review.suggestions:
            logger.info(f"Suggestion to apply: {suggestion}")
        
        return plan