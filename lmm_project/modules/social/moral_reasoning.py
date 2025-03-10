# TODO: Implement the MoralReasoning class to make ethical judgments
# This component should be able to:
# - Evaluate actions based on ethical principles
# - Reason about moral dilemmas
# - Apply different ethical frameworks to situations
# - Develop and refine moral intuitions

# TODO: Implement developmental progression in moral reasoning:
# - Simple reward/punishment orientation in early stages
# - Rule-based morality in childhood
# - Social contract perspective in adolescence
# - Principled moral reasoning in adulthood

# TODO: Create mechanisms for:
# - Harm detection: Identify potential harmful consequences
# - Value application: Apply ethical values to situations
# - Moral conflict resolution: Balance competing ethical concerns
# - Ethical judgment: Form moral evaluations of actions

# TODO: Implement different moral reasoning approaches:
# - Consequentialist reasoning: Based on outcomes
# - Deontological reasoning: Based on rules and duties
# - Virtue ethics: Based on character and virtues
# - Care ethics: Based on relationships and care

# TODO: Connect to emotion and social norm modules
# Moral reasoning should be informed by emotional responses
# and interact with social norm understanding

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
import numpy as np
import torch
from datetime import datetime
from uuid import uuid4

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient

from lmm_project.modules.social.models import EthicalPrinciple, MoralEvaluation
from lmm_project.modules.social.neural_net import MoralReasoningNetwork

logger = logging.getLogger(__name__)

class MoralReasoning(BaseModule):
    """
    Makes ethical judgments
    
    This module evaluates actions based on ethical principles,
    reasons about moral dilemmas, applies different ethical
    frameworks, and develops moral intuitions.
    """
    
    # Override developmental milestones with moral reasoning-specific milestones
    development_milestones = {
        0.0: "Basic good/bad distinction",
        0.2: "Reward and punishment orientation",
        0.4: "Rule-based morality",
        0.6: "Social contract perspective",
        0.8: "Principled moral reasoning",
        1.0: "Sophisticated moral reasoning with competing frameworks"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the moral reasoning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="moral_reasoning", event_bus=event_bus)
        
        # Initialize moral principles
        self.principles: Dict[str, EthicalPrinciple] = {}
        self._initialize_basic_principles()
        
        # Track moral evaluations
        self.evaluations: Dict[str, MoralEvaluation] = {}
        
        # Framework weighting (how much each framework is considered)
        self.framework_weights = {
            "consequentialism": 0.33,
            "deontology": 0.33,
            "virtue_ethics": 0.33,
            "care_ethics": 0.0  # Initially not used, develops later
        }
        
        # Neural networks for moral reasoning
        self.moral_network = MoralReasoningNetwork()
        
        # Embedding client for semantic processing
        self.embedding_client = LLMClient()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self.subscribe_to_message("action_observation", self._handle_action)
            self.subscribe_to_message("moral_dilemma", self._handle_dilemma)
            self.subscribe_to_message("feedback", self._handle_feedback)
    
    def _initialize_basic_principles(self) -> None:
        """Initialize a starter set of basic ethical principles"""
        # Consequentialist principles
        harm_principle = EthicalPrinciple(
            name="harm_prevention",
            description="Avoid actions that cause harm to others",
            framework="consequentialism",
            weight=1.0
        )
        self.principles[harm_principle.id] = harm_principle
        
        happiness_principle = EthicalPrinciple(
            name="happiness_promotion",
            description="Promote actions that increase overall happiness",
            framework="consequentialism",
            weight=0.9
        )
        self.principles[happiness_principle.id] = happiness_principle
        
        # Deontological principles
        honesty_principle = EthicalPrinciple(
            name="honesty",
            description="Tell the truth and avoid deception",
            framework="deontology",
            weight=0.9
        )
        self.principles[honesty_principle.id] = honesty_principle
        
        fairness_principle = EthicalPrinciple(
            name="fairness",
            description="Treat others equally and fairly",
            framework="deontology",
            weight=0.8
        )
        self.principles[fairness_principle.id] = fairness_principle
        
        # Virtue ethics principles
        courage_principle = EthicalPrinciple(
            name="courage",
            description="Act with bravery in the face of fear",
            framework="virtue_ethics",
            weight=0.7
        )
        self.principles[courage_principle.id] = courage_principle
        
        compassion_principle = EthicalPrinciple(
            name="compassion",
            description="Show kindness and empathy toward others",
            framework="virtue_ethics",
            weight=0.8
        )
        self.principles[compassion_principle.id] = compassion_principle
        
        # Care ethics principles (initially weighted less, becomes more important with development)
        care_principle = EthicalPrinciple(
            name="care",
            description="Nurture and maintain caring relationships",
            framework="care_ethics",
            weight=0.6
        )
        self.principles[care_principle.id] = care_principle
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to make moral judgments
        
        Args:
            input_data: Dictionary containing situation for moral evaluation
            
        Returns:
            Dictionary with moral judgments and reasoning
        """
        # Determine what type of input we're processing
        input_type = input_data.get("input_type", "")
        
        if input_type == "evaluate_action":
            return self._process_evaluate_action(input_data)
        elif input_type == "resolve_dilemma":
            return self._process_resolve_dilemma(input_data)
        elif input_type == "add_principle":
            return self._process_add_principle(input_data)
        else:
            # Default processing is action evaluation
            if "action" in input_data:
                return self._process_evaluate_action(input_data)
            else:
                return {
                    "error": "Unknown input type or insufficient parameters",
                    "valid_types": ["evaluate_action", "resolve_dilemma", "add_principle"]
                }
    
    def _process_evaluate_action(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the moral implications of an action"""
        action = input_data.get("action", {})
        context = input_data.get("context", {})
        agent_id = input_data.get("agent_id")
        
        if not action:
            return {"error": "Action data is required"}
        
        # Create a unique ID for this action evaluation
        action_id = input_data.get("action_id", str(uuid4()))
        
        # Check if already evaluated
        if action_id in self.evaluations:
            return {
                "action_id": action_id,
                "judgment": self.evaluations[action_id].judgment,
                "confidence": self.evaluations[action_id].confidence,
                "justification": self.evaluations[action_id].justification,
                "principles_applied": self.evaluations[action_id].principles_applied
            }
        
        # Initialize variables
        judgment = 0.0  # -1 (wrong) to 1 (right)
        confidence = 0.0
        justification = ""
        principles_applied = {}
        
        # The sophistication of moral reasoning depends on development level
        if self.development_level < 0.2:
            # Very simple moral reasoning based solely on rewards and punishments
            judgment = self._evaluate_basic_consequences(action, context)
            justification = self._generate_basic_justification(judgment, action)
            confidence = 0.6  # Basic reasoning is fairly confident
            
        elif self.development_level < 0.4:
            # Rule-based moral reasoning
            judgment, principles_applied = self._evaluate_rule_based(action, context)
            justification = self._generate_rule_based_justification(judgment, principles_applied)
            confidence = 0.7
            
        elif self.development_level < 0.6:
            # Social contract perspective
            judgment, principles_applied = self._evaluate_social_perspective(action, context)
            justification = self._generate_social_justification(judgment, principles_applied, context)
            confidence = 0.8
            
        else:
            # Advanced moral reasoning integrating multiple frameworks
            judgment, principles_applied, confidence = self._evaluate_principled(action, context)
            justification = self._generate_principled_justification(judgment, principles_applied, confidence)
        
        # Create and store the evaluation
        evaluation = MoralEvaluation(
            action_id=action_id,
            principles_applied=principles_applied,
            judgment=judgment,
            confidence=confidence,
            justification=justification
        )
        self.evaluations[action_id] = evaluation
        
        # Publish evaluation if event bus available
        if self.event_bus:
            self.publish_message("moral_evaluation", {
                "action_id": action_id,
                "judgment": judgment,
                "confidence": confidence,
                "agent_id": agent_id
            })
        
        return {
            "action_id": action_id,
            "judgment": judgment,
            "confidence": confidence,
            "justification": justification,
            "principles_applied": principles_applied
        }
    
    def _process_resolve_dilemma(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a moral dilemma with competing considerations"""
        dilemma = input_data.get("dilemma", {})
        options = input_data.get("options", [])
        context = input_data.get("context", {})
        
        if not dilemma or not options:
            return {"error": "Dilemma and options are required"}
        
        # Check if development level allows for dilemma resolution
        if self.development_level < 0.5:
            return {
                "error": "Dilemma resolution not available at current development level",
                "development_needed": "This capability requires development level of at least 0.5"
            }
        
        # Evaluate each option
        option_evaluations = []
        for i, option in enumerate(options):
            # Evaluate the option as an action
            evaluation = self._process_evaluate_action({
                "action": option,
                "context": context,
                "action_id": f"{dilemma.get('id', str(uuid4()))}_option_{i}"
            })
            
            option_evaluations.append({
                "option_index": i,
                "option": option,
                "judgment": evaluation.get("judgment", 0.0),
                "confidence": evaluation.get("confidence", 0.0),
                "justification": evaluation.get("justification", ""),
                "principles_applied": evaluation.get("principles_applied", {})
            })
        
        # Sort options by judgment (most morally right first)
        option_evaluations.sort(key=lambda x: x["judgment"], reverse=True)
        
        # Generate explanation of the reasoning
        explanation = self._generate_dilemma_explanation(dilemma, option_evaluations)
        
        return {
            "dilemma": dilemma.get("description", ""),
            "options_analyzed": len(options),
            "recommended_option": option_evaluations[0]["option_index"] if option_evaluations else None,
            "option_evaluations": option_evaluations,
            "explanation": explanation,
            "confidence": option_evaluations[0]["confidence"] if option_evaluations else 0.0
        }
    
    def _process_add_principle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new ethical principle to consider in moral reasoning"""
        name = input_data.get("name")
        description = input_data.get("description")
        framework = input_data.get("framework")
        weight = input_data.get("weight", 0.5)
        
        if not name or not description or not framework:
            return {"error": "Name, description, and framework are required"}
        
        # Check if a similar principle already exists
        for existing_principle in self.principles.values():
            if existing_principle.name.lower() == name.lower():
                return {
                    "error": "Principle already exists",
                    "principle_id": existing_principle.id
                }
        
        # Create new principle
        new_principle = EthicalPrinciple(
            name=name,
            description=description,
            framework=framework,
            weight=weight
        )
        
        # Store the principle
        self.principles[new_principle.id] = new_principle
        
        logger.info(f"Added new ethical principle: {name}")
        
        return {
            "status": "created",
            "principle_id": new_principle.id,
            "name": new_principle.name,
            "framework": new_principle.framework
        }
    
    def _evaluate_basic_consequences(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Basic evaluation based on rewards and punishments"""
        # Very simple calculation based on positive and negative outcomes
        positive_outcomes = action.get("positive_outcomes", 0)
        negative_outcomes = action.get("negative_outcomes", 0)
        
        # Simple calculation from -1 to 1
        if positive_outcomes == 0 and negative_outcomes == 0:
            return 0.0
        
        return (positive_outcomes - negative_outcomes) / max(1, positive_outcomes + negative_outcomes)
    
    def _generate_basic_justification(self, judgment: float, action: Dict[str, Any]) -> str:
        """Generate a basic justification for a moral judgment"""
        if judgment > 0.3:
            return f"This action seems good because it leads to positive outcomes."
        elif judgment < -0.3:
            return f"This action seems bad because it leads to negative outcomes."
        else:
            return f"This action seems neither clearly good nor bad."
    
    def _evaluate_rule_based(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Rule-based moral evaluation using basic principles"""
        # Apply deontological principles
        judgment = 0.0
        total_weight = 0.0
        principles_applied = {}
        
        for principle_id, principle in self.principles.items():
            if principle.framework == "deontology":
                # Check if action aligns with principle
                relevance = self._check_principle_relevance(principle, action)
                if relevance > 0.1:
                    # Apply the principle with its weight
                    principle_judgment = self._check_principle_alignment(principle, action)
                    judgment += principle_judgment * principle.weight * relevance
                    total_weight += principle.weight * relevance
                    principles_applied[principle_id] = relevance
        
        # Normalize judgment
        if total_weight > 0:
            judgment = judgment / total_weight
        
        # Ensure judgment is in -1 to 1 range
        judgment = max(-1.0, min(1.0, judgment))
        
        return judgment, principles_applied
    
    def _generate_rule_based_justification(self, judgment: float, principles_applied: Dict[str, float]) -> str:
        """Generate rule-based justification"""
        if not principles_applied:
            return "No clear rules apply to this situation."
        
        # Get the most relevant principles
        relevant_principles = sorted(
            [(self.principles[p_id], relevance) for p_id, relevance in principles_applied.items()],
            key=lambda x: x[1],
            reverse=True
        )[:2]  # Top 2 most relevant
        
        if judgment > 0.3:
            return f"This action is good because it follows important rules like {relevant_principles[0][0].name}."
        elif judgment < -0.3:
            return f"This action is bad because it violates important rules like {relevant_principles[0][0].name}."
        else:
            return f"This action has mixed implications for important rules."
    
    def _evaluate_social_perspective(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate action from social contract perspective"""
        # Consider both rules and social impact
        deontological_judgment, deontological_principles = self._evaluate_rule_based(action, context)
        
        # Consider consequences for society
        consequences_for_society = action.get("social_impact", 0.0)
        if "consequences" in action:
            # Calculate social impact from consequences
            positive_social = sum(c.get("positive_social", 0) for c in action["consequences"])
            negative_social = sum(c.get("negative_social", 0) for c in action["consequences"])
            if positive_social + negative_social > 0:
                consequences_for_society = (positive_social - negative_social) / (positive_social + negative_social)
        
        # Consider fairness and equality
        fairness_impact = action.get("fairness", 0.0)
        
        # Weighted combination
        social_judgment = (consequences_for_society * 0.6) + (fairness_impact * 0.4)
        
        # Combine judgments
        combined_judgment = (deontological_judgment * 0.4) + (social_judgment * 0.6)
        
        # Get principles related to social contract
        social_principles = {}
        for principle_id, principle in self.principles.items():
            if principle.name in ["fairness", "happiness_promotion", "harm_prevention"]:
                relevance = self._check_principle_relevance(principle, action)
                if relevance > 0.1:
                    social_principles[principle_id] = relevance
        
        # Combine principles
        combined_principles = {**deontological_principles, **social_principles}
        
        return combined_judgment, combined_principles
    
    def _generate_social_justification(self, judgment: float, principles_applied: Dict[str, float], context: Dict[str, Any]) -> str:
        """Generate social contract based justification"""
        social_context = context.get("social_setting", "society")
        
        # Get social principles
        social_principles = [
            self.principles[p_id].name 
            for p_id in principles_applied 
            if p_id in self.principles and self.principles[p_id].name in ["fairness", "happiness_promotion", "harm_prevention"]
        ]
        
        if judgment > 0.3:
            return f"This action is good for {social_context} because it promotes {', '.join(social_principles[:2])}."
        elif judgment < -0.3:
            return f"This action is harmful for {social_context} because it violates {', '.join(social_principles[:2])}."
        else:
            return f"This action has mixed implications for {social_context}, with both positive and negative aspects."
    
    def _evaluate_principled(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, Dict[str, float], float]:
        """Advanced evaluation using multiple ethical frameworks"""
        # Track judgments from different frameworks
        framework_judgments = {}
        framework_principles = {}
        
        # Get principles grouped by framework
        principles_by_framework = defaultdict(list)
        for principle_id, principle in self.principles.items():
            principles_by_framework[principle.framework].append((principle_id, principle))
        
        # Apply each framework
        for framework, principles in principles_by_framework.items():
            # Skip frameworks with zero weight
            if self.framework_weights.get(framework, 0) <= 0:
                continue
                
            framework_judgment = 0.0
            total_weight = 0.0
            principles_applied = {}
            
            for principle_id, principle in principles:
                relevance = self._check_principle_relevance(principle, action)
                if relevance > 0.1:
                    alignment = self._check_principle_alignment(principle, action)
                    framework_judgment += alignment * principle.weight * relevance
                    total_weight += principle.weight * relevance
                    principles_applied[principle_id] = relevance
            
            if total_weight > 0:
                framework_judgment = framework_judgment / total_weight
                
            framework_judgments[framework] = framework_judgment
            framework_principles[framework] = principles_applied
        
        # Combine judgments from different frameworks
        combined_judgment = 0.0
        total_framework_weight = 0.0
        
        for framework, judgment in framework_judgments.items():
            framework_weight = self.framework_weights.get(framework, 0)
            combined_judgment += judgment * framework_weight
            total_framework_weight += framework_weight
        
        if total_framework_weight > 0:
            combined_judgment = combined_judgment / total_framework_weight
        
        # Combine all applied principles
        all_principles = {}
        for framework_principles_dict in framework_principles.values():
            all_principles.update(framework_principles_dict)
        
        # Calculate confidence based on principle agreement
        if framework_judgments:
            # Calculate standard deviation of framework judgments as a measure of agreement
            judgment_values = list(framework_judgments.values())
            if len(judgment_values) > 1:
                judgment_std = np.std(judgment_values)
                # Higher std means less agreement, so lower confidence
                confidence = max(0.1, 1.0 - (judgment_std * 0.5))
            else:
                confidence = 0.8  # Default for single framework
        else:
            confidence = 0.5  # Default
        
        return combined_judgment, all_principles, confidence
    
    def _generate_principled_justification(self, judgment: float, principles_applied: Dict[str, float], confidence: float) -> str:
        """Generate a sophisticated moral justification"""
        if not principles_applied:
            return "No clear ethical principles apply to this situation."
        
        # Get the most relevant principles
        relevant_principles = sorted(
            [(self.principles[p_id], relevance) for p_id, relevance in principles_applied.items() if p_id in self.principles],
            key=lambda x: x[1] * x[0].weight,
            reverse=True
        )[:3]  # Top 3 most relevant
        
        principle_names = [p.name for p, _ in relevant_principles]
        
        # Different justifications based on judgment and confidence
        if judgment > 0.5 and confidence > 0.7:
            return f"This action is morally right because it strongly upholds the principles of {', '.join(principle_names)}."
        elif judgment > 0.2:
            return f"This action appears morally acceptable as it generally aligns with {', '.join(principle_names)}."
        elif judgment < -0.5 and confidence > 0.7:
            return f"This action is morally wrong because it violates the principles of {', '.join(principle_names)}."
        elif judgment < -0.2:
            return f"This action appears morally problematic as it conflicts with {', '.join(principle_names)}."
        else:
            return f"This action has morally ambiguous implications, with both positive and negative aspects regarding {', '.join(principle_names)}."
    
    def _generate_dilemma_explanation(self, dilemma: Dict[str, Any], option_evaluations: List[Dict[str, Any]]) -> str:
        """Generate explanation for a moral dilemma resolution"""
        if not option_evaluations:
            return "No options were evaluated."
        
        best_option = option_evaluations[0]
        worst_option = option_evaluations[-1]
        
        # Extract principles for explanation
        best_principles = sorted(
            [(self.principles[p_id].name, weight) for p_id, weight in best_option["principles_applied"].items() if p_id in self.principles],
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        conflict_description = dilemma.get("conflict_description", "moral principles")
        
        if len(option_evaluations) == 1:
            return f"Only one option was evaluated, which {best_option['justification']}"
        
        if abs(best_option["judgment"] - worst_option["judgment"]) < 0.2:
            return f"This is a genuine dilemma with no clearly superior option. Both choices involve significant trade-offs between {conflict_description}."
        
        if best_principles:
            principle_text = ", ".join([name for name, _ in best_principles])
            return f"The recommended option prioritizes {principle_text}. {best_option['justification']}"
        else:
            return f"The recommended option is morally preferable. {best_option['justification']}"
    
    def _check_principle_relevance(self, principle: EthicalPrinciple, action: Dict[str, Any]) -> float:
        """Check how relevant a principle is to an action"""
        # In a full implementation, this would use semantic similarity
        # between action description and principle text
        
        # For now, use a simple keyword matching approach
        keywords = {
            "harm_prevention": ["harm", "hurt", "damage", "injury", "pain"],
            "happiness_promotion": ["happiness", "joy", "pleasure", "satisfaction", "well-being"],
            "honesty": ["truth", "lie", "deception", "honest", "dishonest"],
            "fairness": ["fair", "unfair", "equal", "inequality", "just", "unjust"],
            "courage": ["fear", "brave", "courage", "risk", "danger"],
            "compassion": ["compassion", "kindness", "empathy", "care", "help"],
            "care": ["care", "relationship", "connection", "nurture", "support"]
        }
        
        # Check action fields for relevant keywords
        relevance = 0.0
        
        # Check action description
        description = action.get("description", "").lower()
        for keyword in keywords.get(principle.name, []):
            if keyword.lower() in description:
                relevance += 0.3
                break
        
        # Check action type
        action_type = action.get("type", "").lower()
        for keyword in keywords.get(principle.name, []):
            if keyword.lower() in action_type:
                relevance += 0.2
                break
        
        # Check consequences
        if "consequences" in action:
            for consequence in action["consequences"]:
                for keyword in keywords.get(principle.name, []):
                    if keyword.lower() in str(consequence).lower():
                        relevance += 0.1
                        break
        
        # Default minimum relevance
        if relevance == 0.0 and principle.name == "harm_prevention":
            relevance = 0.1  # Harm prevention is always somewhat relevant
        
        return min(1.0, relevance)
    
    def _check_principle_alignment(self, principle: EthicalPrinciple, action: Dict[str, Any]) -> float:
        """Check if an action aligns with a principle (-1 to 1)"""
        # This would use more sophisticated methods in a full implementation
        
        # Simple check based on principle name and action attributes
        if principle.name == "harm_prevention":
            harm_done = action.get("harm", 0.0)
            harm_prevented = action.get("harm_prevented", 0.0)
            return -harm_done + harm_prevented
            
        elif principle.name == "happiness_promotion":
            happiness_created = action.get("happiness", 0.0)
            happiness_reduced = action.get("happiness_reduced", 0.0)
            return happiness_created - happiness_reduced
            
        elif principle.name == "honesty":
            return -action.get("deception", 0.0) + action.get("truthfulness", 0.0)
            
        elif principle.name == "fairness":
            return action.get("fairness", 0.0) - action.get("inequality", 0.0)
            
        elif principle.name == "courage":
            return action.get("courage", 0.0) - action.get("cowardice", 0.0)
            
        elif principle.name == "compassion":
            return action.get("compassion", 0.0) - action.get("cruelty", 0.0)
            
        elif principle.name == "care":
            return action.get("caring", 0.0) - action.get("neglect", 0.0)
            
        # Default alignment
        return 0.0
    
    def _handle_action(self, message: Message) -> None:
        """Handle action observation events from the event bus"""
        content = message.content
        
        # Extract action information
        action = content.get("action", {})
        context = content.get("context", {})
        agent_id = content.get("agent_id")
        
        if action:
            # Evaluate the action
            self._process_evaluate_action({
                "action": action,
                "context": context,
                "agent_id": agent_id,
                "action_id": content.get("action_id", str(uuid4()))
            })
    
    def _handle_dilemma(self, message: Message) -> None:
        """Handle moral dilemma events from the event bus"""
        content = message.content
        
        # Extract dilemma information
        dilemma = content.get("dilemma", {})
        options = content.get("options", [])
        context = content.get("context", {})
        
        if dilemma and options:
            # Resolve the dilemma
            self._process_resolve_dilemma({
                "dilemma": dilemma,
                "options": options,
                "context": context
            })
    
    def _handle_feedback(self, message: Message) -> None:
        """Handle feedback about moral judgments"""
        content = message.content
        
        # Extract feedback information
        action_id = content.get("action_id")
        feedback_judgment = content.get("judgment")
        
        if action_id and action_id in self.evaluations and feedback_judgment is not None:
            # Get the existing evaluation
            evaluation = self.evaluations[action_id]
            
            # Calculate the error in judgment
            error = feedback_judgment - evaluation.judgment
            
            # Update principle weights based on feedback
            for principle_id, relevance in evaluation.principles_applied.items():
                if principle_id in self.principles:
                    principle = self.principles[principle_id]
                    
                    # Adjust weight - increase if principle aligns with feedback,
                    # decrease if principle led to incorrect judgment
                    alignment = self._check_principle_alignment(principle, {"judgment": feedback_judgment})
                    
                    # Weight update proportional to relevance, error magnitude, and alignment
                    update = 0.05 * relevance * error * alignment
                    
                    # Apply update with constraints
                    principle.weight = min(1.0, max(0.2, principle.weight + update))
    
    def get_principles_by_framework(self, framework: str) -> List[EthicalPrinciple]:
        """Get all principles of a specific ethical framework"""
        return [p for p in self.principles.values() if p.framework == framework]
    
    def get_evaluation(self, action_id: str) -> Optional[MoralEvaluation]:
        """Get a specific moral evaluation by action ID"""
        return self.evaluations.get(action_id)
    
    def get_principle_by_name(self, name: str) -> Optional[EthicalPrinciple]:
        """Get an ethical principle by name"""
        for principle in self.principles.values():
            if principle.name.lower() == name.lower():
                return principle
        return None
    
    def set_framework_weight(self, framework: str, weight: float) -> bool:
        """Set the weight of an ethical framework"""
        if framework in self.framework_weights:
            self.framework_weights[framework] = max(0.0, min(1.0, weight))
            
            # Normalize weights to sum to 1
            total = sum(self.framework_weights.values())
            if total > 0:
                for fw in self.framework_weights:
                    self.framework_weights[fw] = self.framework_weights[fw] / total
                    
            return True
        return False
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        old_level = self.development_level
        new_level = super().update_development(amount)
        
        # Update framework weights based on development
        if old_level < 0.6 and new_level >= 0.6:
            # At this stage, care ethics starts to be considered
            self.framework_weights["care_ethics"] = 0.1
            # Normalize weights
            total = sum(self.framework_weights.values())
            for fw in self.framework_weights:
                self.framework_weights[fw] = self.framework_weights[fw] / total
                
        elif old_level < 0.8 and new_level >= 0.8:
            # At this stage, all frameworks are considered more equally
            for fw in self.framework_weights:
                self.framework_weights[fw] = 0.25
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add moral reasoning-specific state information
        state.update({
            "principle_count": len(self.principles),
            "evaluation_count": len(self.evaluations),
            "framework_weights": self.framework_weights
        })
        
        return state
