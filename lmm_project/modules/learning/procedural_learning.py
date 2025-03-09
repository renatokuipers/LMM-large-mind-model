import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import uuid
import logging
import os
from collections import defaultdict

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.learning.models import ProceduralLearningEvent

logger = logging.getLogger(__name__)

class ProceduralLearning(BaseModule):
    """
    Learning skills and procedures through practice
    
    This module develops procedural knowledge through repetition and practice,
    gradually improving performance on tasks and automating sequences of actions.
    """
    
    # Development milestones for procedural learning
    development_milestones = {
        0.0: "Simple action sequences",
        0.2: "Basic skill coordination",
        0.4: "Skill refinement through practice",
        0.6: "Efficient procedure optimization",
        0.8: "Automated procedural execution",
        1.0: "Complex skill integration"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the procedural learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="procedural_learning",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Skills and procedures being learned
        # {skill_name: {proficiency, practice_count, steps, etc.}}
        self.skills = {}
        
        # Performance metrics for each skill
        self.performance_history = defaultdict(list)
        
        # Developmental parameters
        self.learning_rate = 0.1  # Base rate for skill improvement
        self.forgetting_rate = 0.01  # How quickly skills decay without practice
        self.automation_threshold = 0.8  # Proficiency level for automation
        
        # Adjust parameters based on development level
        self._adjust_for_development()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("skill_practice", self._handle_practice)
            self.subscribe_to_message("skill_performance", self._handle_performance)
    
    def _adjust_for_development(self):
        """Adjust learning mechanisms based on developmental level"""
        # Learning rate increases with development (faster skill acquisition)
        self.learning_rate = 0.1 + (self.development_level * 0.15)
        
        # Forgetting rate decreases with development (better retention)
        self.forgetting_rate = max(0.001, 0.02 - (self.development_level * 0.019))
        
        # Automation threshold decreases with development (easier automation)
        self.automation_threshold = max(0.5, 0.9 - (self.development_level * 0.4))
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for procedural learning
        
        Args:
            input_data: Dictionary containing skill practice information
            
        Returns:
            Dictionary with updated skill proficiency and performance
        """
        operation = input_data.get("operation", "practice")
        
        if operation == "practice":
            return self._practice_skill(input_data)
        elif operation == "learn_sequence":
            return self._learn_sequence(input_data)
        elif operation == "recall_skill":
            return self._recall_skill(input_data)
        elif operation == "check_automation":
            return self._check_automation(input_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "module_id": self.module_id
            }
    
    def _practice_skill(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Practice an existing skill or create a new one"""
        skill_name = input_data.get("skill")
        practice_quality = input_data.get("quality", 0.5)  # How well the practice was performed
        practice_duration = input_data.get("duration", 1.0)  # Duration in minutes
        
        if not skill_name:
            return {"status": "error", "message": "Missing skill name"}
        
        # Get or create skill
        if skill_name not in self.skills:
            # Create new skill
            self.skills[skill_name] = {
                "proficiency": 0.1,  # Starting proficiency
                "practice_count": 0,
                "total_practice_time": 0.0,
                "last_practiced": datetime.now(),
                "automated": False,
                "steps": input_data.get("steps", []),
                "dependencies": input_data.get("dependencies", []),
                "created_at": datetime.now()
            }
        
        skill = self.skills[skill_name]
        
        # Calculate proficiency improvement based on practice quality and duration
        # Apply developmental learning rate and diminishing returns
        current_proficiency = skill["proficiency"]
        
        # Calculate practice effectiveness
        # Higher quality practice with longer duration is more effective
        # Diminishing returns as proficiency increases
        effectiveness = practice_quality * practice_duration * self.learning_rate
        
        # Apply diminishing returns (harder to improve as proficiency increases)
        room_for_improvement = 1.0 - current_proficiency
        improvement = effectiveness * room_for_improvement
        
        # Update skill data
        new_proficiency = min(1.0, current_proficiency + improvement)
        skill["proficiency"] = new_proficiency
        skill["practice_count"] += 1
        skill["total_practice_time"] += practice_duration
        skill["last_practiced"] = datetime.now()
        
        # Check if skill should now be automated
        if new_proficiency >= self.automation_threshold and not skill["automated"]:
            skill["automated"] = True
        
        # Update performance history
        self.performance_history[skill_name].append({
            "timestamp": datetime.now(),
            "proficiency": new_proficiency,
            "practice_quality": practice_quality,
            "improvement": improvement
        })
        
        # Trim history if needed
        if len(self.performance_history[skill_name]) > 100:
            self.performance_history[skill_name] = self.performance_history[skill_name][-100:]
        
        # Create learning event
        event = ProceduralLearningEvent(
            source=input_data.get("source", "practice"),
            content=f"Practice of skill '{skill_name}'",
            skill=skill_name,
            proficiency=new_proficiency,
            practice_count=skill["practice_count"],
            practice_time=practice_duration,
            learning_mode=input_data.get("learning_mode", "explicit"),
            procedure_steps=skill["steps"],
            developmental_level=self.development_level
        )
        
        return {
            "status": "success",
            "skill": skill_name,
            "previous_proficiency": current_proficiency,
            "new_proficiency": new_proficiency,
            "improvement": improvement,
            "practice_count": skill["practice_count"],
            "automated": skill["automated"],
            "learning_event_id": event.id
        }
    
    def _learn_sequence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a new sequence or procedure"""
        skill_name = input_data.get("skill")
        steps = input_data.get("steps", [])
        
        if not skill_name or not steps:
            return {"status": "error", "message": "Missing skill name or steps"}
        
        # Check if skill exists
        if skill_name in self.skills:
            # Update existing skill's steps
            self.skills[skill_name]["steps"] = steps
            action = "updated"
        else:
            # Create new skill with these steps
            self.skills[skill_name] = {
                "proficiency": 0.1,  # Starting proficiency
                "practice_count": 0,
                "total_practice_time": 0.0,
                "last_practiced": datetime.now(),
                "automated": False,
                "steps": steps,
                "dependencies": input_data.get("dependencies", []),
                "created_at": datetime.now()
            }
            action = "created"
        
        # Create learning event
        event = ProceduralLearningEvent(
            source=input_data.get("source", "instruction"),
            content=f"Learning sequence for skill '{skill_name}'",
            skill=skill_name,
            proficiency=self.skills[skill_name]["proficiency"],
            practice_count=self.skills[skill_name]["practice_count"],
            practice_time=0.0,
            learning_mode=input_data.get("learning_mode", "explicit"),
            procedure_steps=steps,
            developmental_level=self.development_level
        )
        
        return {
            "status": "success",
            "skill": skill_name,
            "action": action,
            "step_count": len(steps),
            "proficiency": self.skills[skill_name]["proficiency"],
            "learning_event_id": event.id
        }
    
    def _recall_skill(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recall a learned skill and its steps"""
        skill_name = input_data.get("skill")
        
        if not skill_name:
            return {"status": "error", "message": "Missing skill name"}
        
        if skill_name not in self.skills:
            return {"status": "not_found", "message": f"Skill not found: {skill_name}"}
        
        skill = self.skills[skill_name]
        
        # Apply forgetting based on time since last practice
        if "last_practiced" in skill:
            time_since_practice = (datetime.now() - skill["last_practiced"]).total_seconds() / 86400.0  # days
            forgetting = self.forgetting_rate * time_since_practice
            
            # More automated skills are forgotten more slowly
            if skill["automated"]:
                forgetting *= 0.2
            
            # Apply forgetting
            skill["proficiency"] = max(0.1, skill["proficiency"] - forgetting)
        
        # Calculate recall quality based on proficiency
        # Add some randomness to simulate variability in recall
        recall_noise = np.random.normal(0, 0.1)  # Mean 0, std 0.1
        recall_quality = min(1.0, max(0.0, skill["proficiency"] + recall_noise))
        
        # Determine which steps are recalled correctly
        recalled_steps = []
        missed_steps = []
        
        for i, step in enumerate(skill["steps"]):
            # Higher proficiency means better recall
            # Steps are easier to forget as sequence length increases
            step_recall_prob = recall_quality * (1.0 - 0.01 * i)
            
            if np.random.random() < step_recall_prob:
                recalled_steps.append(step)
            else:
                missed_steps.append(step)
        
        # Update last practiced timestamp (recall is a form of practice)
        skill["last_practiced"] = datetime.now()
        
        return {
            "status": "success",
            "skill": skill_name,
            "proficiency": skill["proficiency"],
            "recall_quality": recall_quality,
            "recalled_steps": recalled_steps,
            "missed_steps": missed_steps,
            "total_steps": len(skill["steps"]),
            "recall_success_rate": len(recalled_steps) / max(1, len(skill["steps"])),
            "automated": skill["automated"]
        }
    
    def _check_automation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a skill is automated and can be performed without conscious effort"""
        skill_name = input_data.get("skill")
        
        if not skill_name:
            return {"status": "error", "message": "Missing skill name"}
        
        if skill_name not in self.skills:
            return {"status": "not_found", "message": f"Skill not found: {skill_name}"}
        
        skill = self.skills[skill_name]
        
        # A skill is considered automated if:
        # 1. Proficiency is above automation threshold
        # 2. It has been practiced sufficiently
        # 3. Time since last practice isn't too long
        
        proficiency_check = skill["proficiency"] >= self.automation_threshold
        practice_check = skill["practice_count"] >= 5 + (len(skill["steps"]) * 2)
        
        recency_check = True
        if "last_practiced" in skill:
            days_since_practice = (datetime.now() - skill["last_practiced"]).total_seconds() / 86400.0
            recency_check = days_since_practice < (7.0 + (skill["proficiency"] * 30.0))
        
        # Update automation status
        is_automated = proficiency_check and practice_check and recency_check
        skill["automated"] = is_automated
        
        # Calculate cognitive load reduction from automation
        if is_automated:
            # More proficient = less cognitive load
            cognitive_load = max(0.1, 1.0 - skill["proficiency"])
        else:
            # Non-automated skills have high cognitive load
            cognitive_load = 0.5 + (0.5 * (1.0 - skill["proficiency"]))
        
        return {
            "status": "success",
            "skill": skill_name,
            "automated": is_automated,
            "proficiency": skill["proficiency"],
            "practice_count": skill["practice_count"],
            "cognitive_load": cognitive_load,
            "automation_checks": {
                "proficiency_sufficient": proficiency_check,
                "practice_sufficient": practice_check,
                "recency_sufficient": recency_check
            }
        }
    
    def _handle_practice(self, message):
        """Handle skill practice events"""
        if not message.content:
            return
            
        practice_data = message.content
        
        # Process the practice event
        if "skill" in practice_data:
            self._practice_skill(practice_data)
    
    def _handle_performance(self, message):
        """Handle skill performance feedback"""
        if not message.content:
            return
            
        performance_data = message.content
        
        # Process the performance data
        if "skill" in performance_data and "quality" in performance_data:
            # Use performance quality to adjust proficiency
            self._practice_skill({
                "skill": performance_data["skill"],
                "quality": performance_data["quality"],
                "duration": performance_data.get("duration", 0.5),
                "source": "performance"
            })
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # If development changed significantly, adjust parameters
        if abs(new_level - previous_level) >= 0.05:
            self._adjust_for_development()
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        base_state = super().get_state()
        
        # Calculate skill statistics
        skill_count = len(self.skills)
        automated_count = sum(1 for skill in self.skills.values() if skill.get("automated", False))
        avg_proficiency = 0.0
        if skill_count > 0:
            avg_proficiency = sum(skill["proficiency"] for skill in self.skills.values()) / skill_count
        
        # Add procedural learning specific state
        module_state = {
            "skill_count": skill_count,
            "automated_skills": automated_count,
            "average_proficiency": avg_proficiency,
            "learning_rate": self.learning_rate,
            "forgetting_rate": self.forgetting_rate,
            "automation_threshold": self.automation_threshold
        }
        
        base_state.update(module_state)
        return base_state
