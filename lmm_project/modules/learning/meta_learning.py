import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime
import uuid
import logging
import os
from collections import defaultdict

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.learning.models import MetaLearningEvent, LearningStrategy

logger = logging.getLogger(__name__)

class MetaLearning(BaseModule):
    """
    Learning how to learn more effectively
    
    This module develops strategies for learning, monitors learning effectiveness,
    and optimizes the application of learning techniques across domains.
    """
    
    # Development milestones for meta-learning
    development_milestones = {
        0.0: "Basic learning reflection",
        0.2: "Simple strategy selection",
        0.4: "Learning strategy adaptation",
        0.6: "Strategic knowledge transfer",
        0.8: "Learning efficiency optimization",
        1.0: "Advanced meta-cognitive control"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the meta-learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="meta_learning",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Learning strategies repository
        self.strategies = {}
        
        # Learning effectiveness by domain
        self.domain_effectiveness = {}
        
        # Strategy usage history
        self.strategy_history = []
        
        # Learning rate adjustment factor (meta-learning rate)
        self.meta_learning_rate = 0.05
        
        # Initialize with basic strategies
        self._initialize_basic_strategies()
        
        # Adjust parameters based on development level
        self._adjust_for_development()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("learning_outcome", self._handle_learning_outcome)
            self.subscribe_to_message("strategy_effectiveness", self._handle_strategy_effectiveness)
    
    def _initialize_basic_strategies(self):
        """Initialize a set of basic learning strategies"""
        basic_strategies = [
            {
                "name": "repetition",
                "description": "Learn through repeated exposure and practice",
                "effectiveness": 0.5,
                "cognitive_load": 0.3,
                "min_developmental_level": 0.0,
                "applicable_domains": ["procedural", "factual", "language"]
            },
            {
                "name": "association",
                "description": "Learn by associating new information with known concepts",
                "effectiveness": 0.6,
                "cognitive_load": 0.4,
                "min_developmental_level": 0.1,
                "applicable_domains": ["semantic", "factual", "conceptual"]
            },
            {
                "name": "trial_and_error",
                "description": "Learn through experimentation and feedback",
                "effectiveness": 0.5,
                "cognitive_load": 0.5,
                "min_developmental_level": 0.0,
                "applicable_domains": ["procedural", "problem-solving"]
            },
            {
                "name": "chunking",
                "description": "Group information into meaningful chunks",
                "effectiveness": 0.7,
                "cognitive_load": 0.6,
                "min_developmental_level": 0.3,
                "applicable_domains": ["memory", "factual", "conceptual"]
            },
        ]
        
        # Create strategy objects
        for strategy_data in basic_strategies:
            strategy = LearningStrategy(
                name=strategy_data["name"],
                description=strategy_data["description"],
                effectiveness=strategy_data["effectiveness"],
                cognitive_load=strategy_data["cognitive_load"],
                min_developmental_level=strategy_data["min_developmental_level"],
                applicable_domains=strategy_data["applicable_domains"],
                created_at=datetime.now(),
                usage_count=0,
                success_rate=0.5
            )
            self.strategies[strategy.id] = strategy
    
    def _adjust_for_development(self):
        """Adjust capabilities based on developmental level"""
        # Meta-learning rate increases with development
        self.meta_learning_rate = 0.05 + (self.development_level * 0.1)
        
        # At higher development levels, unlock more advanced strategies
        if self.development_level >= 0.4 and not any(s.name == "comparison" for s in self.strategies.values()):
            self._add_advanced_strategies()
    
    def _add_advanced_strategies(self):
        """Add more advanced learning strategies that unlock at higher development levels"""
        advanced_strategies = [
            {
                "name": "comparison",
                "description": "Learn by comparing similarities and differences",
                "effectiveness": 0.7,
                "cognitive_load": 0.6,
                "min_developmental_level": 0.4,
                "applicable_domains": ["conceptual", "analytical", "relational"]
            },
            {
                "name": "elaboration",
                "description": "Expand on information by adding details or connections",
                "effectiveness": 0.8,
                "cognitive_load": 0.7,
                "min_developmental_level": 0.5,
                "applicable_domains": ["conceptual", "factual", "semantic"]
            },
            {
                "name": "self_explanation",
                "description": "Explain concepts to oneself to deepen understanding",
                "effectiveness": 0.8,
                "cognitive_load": 0.7,
                "min_developmental_level": 0.6,
                "applicable_domains": ["conceptual", "procedural", "analytical"]
            },
            {
                "name": "interleaving",
                "description": "Alternate between different topics or skills during learning",
                "effectiveness": 0.8,
                "cognitive_load": 0.8,
                "min_developmental_level": 0.7,
                "applicable_domains": ["procedural", "problem-solving", "motor"]
            },
        ]
        
        # Create strategy objects
        for strategy_data in advanced_strategies:
            # Only add if development level is sufficient
            if self.development_level >= strategy_data["min_developmental_level"]:
                strategy = LearningStrategy(
                    name=strategy_data["name"],
                    description=strategy_data["description"],
                    effectiveness=strategy_data["effectiveness"],
                    cognitive_load=strategy_data["cognitive_load"],
                    min_developmental_level=strategy_data["min_developmental_level"],
                    applicable_domains=strategy_data["applicable_domains"],
                    created_at=datetime.now(),
                    usage_count=0,
                    success_rate=0.5
                )
                self.strategies[strategy.id] = strategy
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for meta-learning operations
        
        Args:
            input_data: Dictionary containing meta-learning parameters
            
        Returns:
            Dictionary with meta-learning results
        """
        operation = input_data.get("operation", "select_strategy")
        
        if operation == "select_strategy":
            return self._select_strategy(input_data)
        elif operation == "evaluate_outcome":
            return self._evaluate_learning_outcome(input_data)
        elif operation == "create_strategy":
            return self._create_learning_strategy(input_data)
        elif operation == "get_strategy":
            return self._get_strategy_details(input_data)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "module_id": self.module_id
            }
    
    def _select_strategy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select an appropriate learning strategy for a given context"""
        domain = input_data.get("domain", "general")
        content_type = input_data.get("content_type", "factual")
        available_cognitive_resources = input_data.get("cognitive_resources", 0.8)
        
        # Filter strategies by developmental level and cognitive resources
        available_strategies = [
            s for s in self.strategies.values()
            if s.min_developmental_level <= self.development_level
            and s.cognitive_load <= available_cognitive_resources
        ]
        
        if not available_strategies:
            return {
                "status": "error",
                "message": "No suitable strategies available",
                "developmental_level": self.development_level
            }
        
        # Calculate strategy scores based on multiple factors
        strategy_scores = {}
        for strategy in available_strategies:
            # Base score is the strategy's effectiveness
            score = strategy.effectiveness
            
            # Bonus if the strategy applies to this domain
            if domain in strategy.applicable_domains:
                score += 0.2
            
            # Bonus for content type match (using domain as proxy)
            if content_type in strategy.applicable_domains:
                score += 0.1
            
            # Success rate influences score (if used before)
            if strategy.usage_count > 0:
                score = (score + strategy.success_rate) / 2
            
            # Efficiency factor (effectiveness/cognitive_load ratio)
            efficiency = strategy.effectiveness / max(0.1, strategy.cognitive_load)
            score = (score + efficiency * 0.3) / 1.3
            
            strategy_scores[strategy.id] = score
        
        # Select best strategy
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        best_strategy = self.strategies[best_strategy_id]
        
        # Increase usage count
        best_strategy.usage_count += 1
        
        # Record in history
        self.strategy_history.append({
            "strategy_id": best_strategy_id,
            "domain": domain,
            "content_type": content_type,
            "cognitive_resources": available_cognitive_resources,
            "timestamp": datetime.now(),
            "score": strategy_scores[best_strategy_id]
        })
        
        # Create meta-learning event
        event = MetaLearningEvent(
            source=input_data.get("source", "meta_learning"),
            content=f"Strategy selection for {domain}/{content_type} learning",
            strategy=best_strategy.name,
            effectiveness=best_strategy.effectiveness,
            applicable_contexts=[domain, content_type],
            target_learning_types=input_data.get("learning_types", [content_type]),
            resource_cost=best_strategy.cognitive_load,
            developmental_level=self.development_level
        )
        
        return {
            "status": "success",
            "selected_strategy": {
                "id": best_strategy_id,
                "name": best_strategy.name,
                "description": best_strategy.description,
                "effectiveness": best_strategy.effectiveness,
                "cognitive_load": best_strategy.cognitive_load
            },
            "domain": domain,
            "content_type": content_type,
            "strategy_score": strategy_scores[best_strategy_id],
            "learning_event_id": event.id
        }
    
    def _evaluate_learning_outcome(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the outcome of a learning strategy application"""
        strategy_id = input_data.get("strategy_id")
        domain = input_data.get("domain", "general")
        success_level = input_data.get("success_level", 0.5)  # 0.0 to 1.0
        
        if not strategy_id or strategy_id not in self.strategies:
            return {"status": "error", "message": "Invalid strategy ID"}
        
        strategy = self.strategies[strategy_id]
        
        # Update success rate using exponential moving average
        if strategy.usage_count <= 1:
            strategy.success_rate = success_level
        else:
            # More weight on recent outcomes at higher development levels
            alpha = 0.2 + (self.development_level * 0.3)
            strategy.success_rate = (alpha * success_level) + ((1 - alpha) * strategy.success_rate)
        
        # Update domain effectiveness
        if domain not in self.domain_effectiveness:
            self.domain_effectiveness[domain] = {}
        
        if strategy_id not in self.domain_effectiveness[domain]:
            self.domain_effectiveness[domain][strategy_id] = {
                "success_sum": 0.0,
                "usage_count": 0
            }
        
        self.domain_effectiveness[domain][strategy_id]["success_sum"] += success_level
        self.domain_effectiveness[domain][strategy_id]["usage_count"] += 1
        
        # Calculate domain-specific effectiveness
        domain_success_rate = (
            self.domain_effectiveness[domain][strategy_id]["success_sum"] / 
            self.domain_effectiveness[domain][strategy_id]["usage_count"]
        )
        
        # At higher development levels, adjust strategy effectiveness based on outcomes
        if self.development_level >= 0.6:
            effectiveness_delta = (success_level - strategy.effectiveness) * self.meta_learning_rate
            strategy.effectiveness = max(0.1, min(1.0, strategy.effectiveness + effectiveness_delta))
        
        return {
            "status": "success",
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "domain": domain,
            "previous_success_rate": strategy.success_rate - ((success_level - strategy.success_rate) * (0.2 + (self.development_level * 0.3))),
            "updated_success_rate": strategy.success_rate,
            "domain_success_rate": domain_success_rate,
            "updated_effectiveness": strategy.effectiveness,
            "meta_learning_rate": self.meta_learning_rate
        }
    
    def _create_learning_strategy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new learning strategy"""
        # Only possible at higher developmental levels
        if self.development_level < 0.5:
            return {
                "status": "error",
                "message": "Creating new strategies requires higher developmental level",
                "current_level": self.development_level,
                "required_level": 0.5
            }
        
        name = input_data.get("name")
        description = input_data.get("description")
        applicable_domains = input_data.get("applicable_domains", [])
        
        if not name or not description:
            return {"status": "error", "message": "Missing strategy name or description"}
        
        # Check if similar strategy already exists
        for strategy in self.strategies.values():
            if strategy.name.lower() == name.lower():
                return {"status": "error", "message": f"Strategy '{name}' already exists"}
        
        # Create new strategy with conservative initial values
        strategy = LearningStrategy(
            name=name,
            description=description,
            effectiveness=input_data.get("effectiveness", 0.5),
            cognitive_load=input_data.get("cognitive_load", 0.6),
            min_developmental_level=input_data.get("min_developmental_level", self.development_level),
            applicable_domains=applicable_domains,
            created_at=datetime.now(),
            usage_count=0,
            success_rate=0.5
        )
        
        # Add to strategies repository
        self.strategies[strategy.id] = strategy
        
        # Create meta-learning event
        event = MetaLearningEvent(
            source=input_data.get("source", "strategy_creation"),
            content=f"Creation of new learning strategy: {name}",
            strategy=name,
            effectiveness=strategy.effectiveness,
            applicable_contexts=applicable_domains,
            target_learning_types=input_data.get("target_learning_types", applicable_domains),
            resource_cost=strategy.cognitive_load,
            developmental_level=self.development_level
        )
        
        return {
            "status": "success",
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "description": strategy.description,
            "effectiveness": strategy.effectiveness,
            "cognitive_load": strategy.cognitive_load,
            "applicable_domains": strategy.applicable_domains,
            "learning_event_id": event.id
        }
    
    def _get_strategy_details(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get details about a specific strategy or list all strategies"""
        strategy_id = input_data.get("strategy_id")
        
        if not strategy_id:
            # Return list of all strategies available at current development level
            available_strategies = [
                {
                    "id": s.id,
                    "name": s.name,
                    "effectiveness": s.effectiveness,
                    "cognitive_load": s.cognitive_load,
                    "applicable_domains": s.applicable_domains,
                    "usage_count": s.usage_count
                }
                for s in self.strategies.values()
                if s.min_developmental_level <= self.development_level
            ]
            
            return {
                "status": "success",
                "strategies": available_strategies,
                "strategy_count": len(available_strategies),
                "developmental_level": self.development_level
            }
        
        # Get specific strategy
        if strategy_id not in self.strategies:
            return {"status": "error", "message": f"Strategy with ID {strategy_id} not found"}
        
        strategy = self.strategies[strategy_id]
        
        # Gather domain-specific effectiveness
        domain_effectiveness = {}
        for domain, strategies in self.domain_effectiveness.items():
            if strategy_id in strategies:
                domain_effectiveness[domain] = (
                    strategies[strategy_id]["success_sum"] / 
                    strategies[strategy_id]["usage_count"]
                )
        
        return {
            "status": "success",
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "effectiveness": strategy.effectiveness,
                "cognitive_load": strategy.cognitive_load,
                "min_developmental_level": strategy.min_developmental_level,
                "applicable_domains": strategy.applicable_domains,
                "usage_count": strategy.usage_count,
                "success_rate": strategy.success_rate,
                "created_at": strategy.created_at.isoformat(),
                "domain_effectiveness": domain_effectiveness
            }
        }
    
    def _handle_learning_outcome(self, message):
        """Handle learning outcome events"""
        if not message.content:
            return
            
        outcome_data = message.content
        
        # Process the learning outcome
        if "strategy_id" in outcome_data and "success_level" in outcome_data:
            self._evaluate_learning_outcome(outcome_data)
    
    def _handle_strategy_effectiveness(self, message):
        """Handle strategy effectiveness feedback"""
        if not message.content:
            return
            
        effectiveness_data = message.content
        
        # Update strategy effectiveness
        if "strategy_id" in effectiveness_data and "effectiveness" in effectiveness_data:
            strategy_id = effectiveness_data["strategy_id"]
            
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                
                # Update effectiveness with a weighted average
                current = strategy.effectiveness
                new_value = effectiveness_data["effectiveness"]
                weight = effectiveness_data.get("weight", 0.3)
                
                strategy.effectiveness = (current * (1 - weight)) + (new_value * weight)
    
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
        
        # Calculate strategy statistics
        strategy_count = len(self.strategies)
        available_strategy_count = sum(
            1 for s in self.strategies.values() 
            if s.min_developmental_level <= self.development_level
        )
        avg_effectiveness = 0.0
        if strategy_count > 0:
            avg_effectiveness = sum(s.effectiveness for s in self.strategies.values()) / strategy_count
        
        # Add meta-learning specific state
        module_state = {
            "strategy_count": strategy_count,
            "available_strategy_count": available_strategy_count,
            "average_effectiveness": avg_effectiveness,
            "meta_learning_rate": self.meta_learning_rate,
            "domain_count": len(self.domain_effectiveness),
            "strategy_usage_history": len(self.strategy_history)
        }
        
        base_state.update(module_state)
        return base_state
