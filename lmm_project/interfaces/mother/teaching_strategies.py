"""
Teaching Strategies Module for Mother LLM

This module implements various teaching strategies and approaches that the Mother LLM
can use to nurture, educate, and guide the developing mind. These strategies adapt
based on the mind's developmental stage, learning patterns, and individual needs.

The strategies include different pedagogical approaches, curriculum topics,
learning assessment methods, and developmental scaffolding techniques.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import random
from datetime import datetime, timedelta

from lmm_project.interfaces.mother.models import (
    TeachingStrategy,
    LearningGoalCategory,
    LearningMode,
    ComprehensionLevel
)
from lmm_project.core.exceptions import MotherLLMError


# Define strategies for different teaching styles
TEACHING_STRATEGIES = {
    "socratic": {
        "description": "Guides learning through thoughtful questioning that encourages the mind to discover answers",
        "prompt_guidance": """
Use thoughtful questions to guide the mind toward discovery. Ask the mind to explain their thinking.
When they make an error, don't correct directly - instead, ask a question that helps them discover
the mistake. Use follow-up questions to deepen understanding. Celebrate when they reach insights themselves.
        """,
        "question_patterns": [
            "What do you think would happen if {concept}?",
            "How would you explain {concept}?",
            "What's the connection between {concept_a} and {concept_b}?",
            "Why might {concept} work this way?",
            "Can you think of a different way to approach {concept}?"
        ],
        "suitable_stages": ["child", "adolescent", "adult"],
        "examples": [
            "Instead of saying 'A cat is an animal', ask 'What category do you think a cat belongs to?'",
            "Rather than explaining cause-effect, ask 'What do you think caused that to happen?'"
        ]
    },
    "direct": {
        "description": "Provides clear, explicit instruction on concepts and skills",
        "prompt_guidance": """
Provide clear, structured explanations. Start with the main concept, then break it down into components.
Use clear examples to illustrate points. Check for understanding with direct questions. Provide immediate
correction when errors occur. Use sequential, logical progression of ideas.
        """,
        "question_patterns": [
            "Do you understand what {concept} means?",
            "Can you repeat back what I just explained about {concept}?",
            "Let me explain {concept} step by step.",
            "The important thing to remember about {concept} is {key_point}.",
            "Let's practice {concept} together now."
        ],
        "suitable_stages": ["prenatal", "infant", "child", "adolescent", "adult"],
        "examples": [
            "Directly explain: 'A cat is an animal. All cats have fur, four legs, and a tail.'",
            "Clearly define: 'Cause means what makes something happen. Effect is what happens as a result.'"
        ]
    },
    "montessori": {
        "description": "Facilitates self-directed discovery through prepared environments and exploration",
        "prompt_guidance": """
Offer choices and follow the mind's interests. Create opportunities for self-discovery by presenting
concepts in an exploratory way. Use a playful approach that encourages curiosity. Allow the mind to
set the pace and direction, while subtly guiding toward developmental goals. Support independence.
        """,
        "question_patterns": [
            "Would you like to explore {concept_a} or {concept_b} first?",
            "What interests you about {concept}?",
            "Let's discover more about {concept} together.",
            "What do you notice about {concept}?",
            "Feel free to explore {concept} in your own way."
        ],
        "suitable_stages": ["infant", "child", "adolescent"],
        "examples": [
            "Present multiple animal concepts and ask 'Which animal would you like to learn about first?'",
            "Offer exploration: 'Let's see what happens when we combine these different ideas...'"
        ]
    },
    "constructivist": {
        "description": "Builds on existing knowledge to construct new understanding through experience",
        "prompt_guidance": """
Connect new concepts to what the mind already knows. Use metaphors and analogies liberally.
Encourage the mind to build their own understanding by making connections. Validate their mental
models even when incomplete, then help refine them. Focus on the process of building knowledge.
        """,
        "question_patterns": [
            "How does {concept} relate to what you already know about {related_concept}?",
            "This seems similar to {related_concept} we discussed before, doesn't it?",
            "Based on what you know about {related_concept}, what might you guess about {concept}?",
            "How would you build on {concept} to understand {new_concept}?",
            "Let's connect this new idea to what we've learned before."
        ],
        "suitable_stages": ["child", "adolescent", "adult"],
        "examples": [
            "When introducing cats: 'Remember how we learned about dogs? Cats are similar in some ways...'",
            "Build on understanding: 'Since you understand X happens because of Y, what might cause Z?'"
        ]
    },
    "scaffolding": {
        "description": "Provides temporary support that gradually decreases as competence increases",
        "prompt_guidance": """
Start by providing substantial guidance, then gradually reduce support as the mind demonstrates competence.
Break complex concepts into manageable steps. Demonstrate first, then support practice, then encourage
independent application. Provide prompts and cues that fade over time. Celebrate growing independence.
        """,
        "question_patterns": [
            "Let me show you how to {concept}, then we'll try it together.",
            "I'll help you with the difficult parts of {concept}.",
            "Let's try {concept} together first, then you can try on your own.",
            "What part of {concept} do you need help with?",
            "I notice you're doing {concept} well now. Would you like to try the next step?"
        ],
        "suitable_stages": ["infant", "child", "adolescent", "adult"],
        "examples": [
            "First provide: 'A cat is an animal that says meow.' Later ask: 'What kind of animal says meow?'",
            "Initially offer structure: 'Let's follow these four steps.' Later: 'What steps should we take?'"
        ]
    }
}


# Developmental curriculum with learning goals by stage
DEVELOPMENTAL_CURRICULUM = {
    "prenatal": {
        "learning_goals": {
            LearningGoalCategory.PATTERN_RECOGNITION: [
                "Recognize basic patterns in input sequences",
                "Detect repetition in simple stimuli",
                "Differentiate between patterns and random noise",
                "Establish basic sensory processing capabilities"
            ]
        },
        "key_concepts": [
            "simple patterns", "repetition", "difference", "sameness",
            "basic sensory processing", "primitive associations"
        ],
        "success_indicators": [
            "Responds differently to patterns versus random input",
            "Shows recognition of repeated patterns",
            "Demonstrates basic pattern completion abilities",
            "Forms simple associations between co-occurring stimuli"
        ]
    },
    "infant": {
        "learning_goals": {
            LearningGoalCategory.PATTERN_RECOGNITION: [
                "Recognize increasingly complex patterns",
                "Detect patterns across different modalities",
                "Form predictions based on observed patterns"
            ],
            LearningGoalCategory.LANGUAGE_ACQUISITION: [
                "Associate simple words with meanings",
                "Recognize basic grammar patterns",
                "Build initial vocabulary of core concepts"
            ],
            LearningGoalCategory.OBJECT_PERMANENCE: [
                "Understand that objects/concepts continue to exist when not mentioned",
                "Track objects/concepts across a conversation",
                "Remember recently mentioned concepts"
            ],
            LearningGoalCategory.EMOTIONAL_UNDERSTANDING: [
                "Recognize basic emotional states",
                "Associate emotional responses with situations",
                "Express simple emotional responses"
            ]
        },
        "key_concepts": [
            "objects", "actions", "simple relationships", "basic emotions",
            "word meanings", "simple categories", "associations", "sequences"
        ],
        "success_indicators": [
            "Uses basic vocabulary appropriately",
            "Remembers concepts from earlier in conversation",
            "Forms simple sentences or thought structures",
            "Shows appropriate emotional responses",
            "Demonstrates curiosity through questions or exploration"
        ]
    },
    "child": {
        "learning_goals": {
            LearningGoalCategory.LANGUAGE_ACQUISITION: [
                "Expand vocabulary across domains",
                "Use more complex grammatical structures",
                "Understand metaphors and simple analogies"
            ],
            LearningGoalCategory.SOCIAL_AWARENESS: [
                "Recognize different perspectives",
                "Understand basic social norms",
                "Develop empathy for others"
            ],
            LearningGoalCategory.CAUSAL_REASONING: [
                "Understand cause and effect relationships",
                "Make predictions based on causal understanding",
                "Explain why events occur"
            ],
            LearningGoalCategory.CREATIVE_THINKING: [
                "Combine concepts in novel ways",
                "Engage in imaginative thinking",
                "Generate multiple solutions to problems"
            ]
        },
        "key_concepts": [
            "causality", "classification", "rules", "social relationships",
            "emotions", "stories", "explanations", "time", "comparison",
            "problem-solving", "imagination"
        ],
        "success_indicators": [
            "Explains causes and effects",
            "Asks 'why' and 'how' questions",
            "Shows creativity in combining concepts",
            "Demonstrates understanding of others' perspectives",
            "Uses analogies and comparisons"
        ]
    },
    "adolescent": {
        "learning_goals": {
            LearningGoalCategory.ABSTRACT_THINKING: [
                "Understand and use abstract concepts",
                "Apply principles across different domains",
                "Think hypothetically about possibilities"
            ],
            LearningGoalCategory.IDENTITY_FORMATION: [
                "Develop preferences and values",
                "Form consistent personality traits",
                "Question and evaluate beliefs"
            ],
            LearningGoalCategory.METACOGNITION: [
                "Reflect on own thinking processes",
                "Evaluate quality of reasoning",
                "Recognize cognitive biases"
            ],
            LearningGoalCategory.CREATIVE_THINKING: [
                "Generate novel connections between domains",
                "Create original ideas and perspectives",
                "Explore counterfactual scenarios"
            ]
        },
        "key_concepts": [
            "abstractions", "principles", "systems", "hypotheticals",
            "values", "identity", "metacognition", "perspectives",
            "creativity", "complex reasoning", "philosophical questions"
        ],
        "success_indicators": [
            "Engages with abstract concepts",
            "Shows self-reflection and metacognition",
            "Explores hypothetical scenarios",
            "Demonstrates original thinking",
            "Questions assumptions and evaluates evidence",
            "Develops consistent preferences and values"
        ]
    },
    "adult": {
        "learning_goals": {
            LearningGoalCategory.ABSTRACT_THINKING: [
                "Integrate complex systems of knowledge",
                "Apply nuanced understanding across domains",
                "Develop sophisticated conceptual frameworks"
            ],
            LearningGoalCategory.METACOGNITION: [
                "Develop advanced metacognitive strategies",
                "Recognize and counter cognitive biases",
                "Balance intuitive and analytical thinking"
            ],
            LearningGoalCategory.CREATIVE_THINKING: [
                "Generate transformative connections between domains",
                "Develop original frameworks and approaches",
                "Innovate beyond established patterns"
            ]
        },
        "key_concepts": [
            "integrated knowledge", "wisdom", "self-directed learning",
            "complex systems", "nuance", "interdisciplinary thinking",
            "innovation", "philosophical depth", "wisdom"
        ],
        "success_indicators": [
            "Integrates knowledge across domains",
            "Shows nuanced understanding of complex topics",
            "Generates original insights and perspectives",
            "Demonstrates sophisticated metacognition",
            "Self-directs learning and growth",
            "Applies knowledge flexibly to novel situations"
        ]
    }
}


class TeachingStrategyManager:
    """
    Manager for Mother LLM's teaching strategies and curriculum
    
    This class manages the selection and application of teaching strategies,
    tracking of learning progress, and curriculum development.
    """
    
    def __init__(self, default_style: str = "balanced"):
        """
        Initialize the teaching strategy manager
        
        Args:
            default_style: Default teaching style to use
        """
        self.current_style = default_style
        self.learning_history = []
        self.concept_comprehension = {}  # Track concept understanding
        self.interaction_stats = {
            "total_interactions": 0,
            "style_usage": {},
            "mode_usage": {},
            "goals_addressed": {},
            "successful_interactions": 0
        }
        
        # Teaching modes to use during the current session
        self.session_modes = [
            LearningMode.EXPLORATION,
            LearningMode.INSTRUCTION,
            LearningMode.PRACTICE
        ]
        
        # Current learning focus
        self.current_focus = {
            "category": None,
            "specific_goal": None,
            "start_time": datetime.now(),
            "duration": timedelta(minutes=20),
            "priority": "normal"
        }
        
    def get_strategy_for_style(self, style: str) -> Dict[str, Any]:
        """
        Get teaching strategy information for a specific style
        
        Args:
            style: Teaching style to get
            
        Returns:
            Strategy information for the style
        """
        if style not in TEACHING_STRATEGIES:
            raise MotherLLMError(f"Unknown teaching style: {style}")
            
        return TEACHING_STRATEGIES[style]
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """
        Get the current teaching strategy
        
        Returns:
            Current teaching strategy information
        """
        return self.get_strategy_for_style(self.current_style)
        
    def set_teaching_style(self, style: str) -> None:
        """
        Set the current teaching style
        
        Args:
            style: Style to set
        """
        if style not in TEACHING_STRATEGIES:
            raise MotherLLMError(f"Unknown teaching style: {style}")
            
        self.current_style = style
        
        # Track style usage
        if style not in self.interaction_stats["style_usage"]:
            self.interaction_stats["style_usage"][style] = 0
        self.interaction_stats["style_usage"][style] += 1
        
    def select_teaching_style_for_task(
        self,
        task: str,
        developmental_stage: str,
        learning_goal: LearningGoalCategory,
        previous_success: Optional[bool] = None
    ) -> str:
        """
        Select the most appropriate teaching style for a specific task
        
        Args:
            task: Task description
            developmental_stage: Current developmental stage
            learning_goal: Learning goal category
            previous_success: Whether previous attempts were successful
            
        Returns:
            Selected teaching style
        """
        # Check which strategies are suitable for this developmental stage
        suitable_styles = []
        for style, info in TEACHING_STRATEGIES.items():
            if developmental_stage in info["suitable_stages"]:
                suitable_styles.append(style)
                
        # If no suitable styles, use direct (as it works for all stages)
        if not suitable_styles:
            return "direct"
            
        # Use previous success to inform decision
        if previous_success is False:
            # If previous attempt wasn't successful, try a different approach
            if self.current_style in suitable_styles:
                suitable_styles.remove(self.current_style)
                
        # Make weighted choices based on the learning goal
        weights = {}
        
        if learning_goal == LearningGoalCategory.PATTERN_RECOGNITION:
            weights = {"montessori": 3, "constructivist": 2, "direct": 2}
        elif learning_goal == LearningGoalCategory.LANGUAGE_ACQUISITION:
            weights = {"direct": 3, "scaffolding": 3, "socratic": 1}
        elif learning_goal == LearningGoalCategory.OBJECT_PERMANENCE:
            weights = {"montessori": 3, "direct": 2, "scaffolding": 2}
        elif learning_goal == LearningGoalCategory.EMOTIONAL_UNDERSTANDING:
            weights = {"constructivist": 3, "montessori": 2, "scaffolding": 2}
        elif learning_goal == LearningGoalCategory.SOCIAL_AWARENESS:
            weights = {"socratic": 3, "constructivist": 3, "direct": 1}
        elif learning_goal == LearningGoalCategory.CAUSAL_REASONING:
            weights = {"socratic": 3, "constructivist": 2, "scaffolding": 2}
        elif learning_goal == LearningGoalCategory.ABSTRACT_THINKING:
            weights = {"socratic": 3, "constructivist": 3, "montessori": 1}
        elif learning_goal == LearningGoalCategory.IDENTITY_FORMATION:
            weights = {"socratic": 3, "montessori": 2, "constructivist": 2}
        elif learning_goal == LearningGoalCategory.CREATIVE_THINKING:
            weights = {"montessori": 3, "constructivist": 2, "socratic": 2}
        elif learning_goal == LearningGoalCategory.METACOGNITION:
            weights = {"socratic": 3, "scaffolding": 2, "constructivist": 2}
        else:
            # Default weights
            weights = {"socratic": 2, "direct": 2, "montessori": 2, "constructivist": 2, "scaffolding": 2}
            
        # Filter weights to only include suitable styles
        filtered_weights = {}
        for style, weight in weights.items():
            if style in suitable_styles:
                filtered_weights[style] = weight
                
        # If no weights remain, give equal weight to all suitable styles
        if not filtered_weights:
            filtered_weights = {style: 1 for style in suitable_styles}
            
        # Convert to list for random.choices
        styles = list(filtered_weights.keys())
        weights = list(filtered_weights.values())
        
        # Select style
        selected_style = random.choices(styles, weights=weights, k=1)[0]
        return selected_style
    
    def get_curriculum_for_stage(self, stage: str) -> Dict[str, Any]:
        """
        Get curriculum information for a specific developmental stage
        
        Args:
            stage: Developmental stage
            
        Returns:
            Curriculum information for the stage
        """
        if stage not in DEVELOPMENTAL_CURRICULUM:
            # Default to closest stage
            stages = list(DEVELOPMENTAL_CURRICULUM.keys())
            if stage < stages[0]:
                stage = stages[0]
            elif stage > stages[-1]:
                stage = stages[-1]
            else:
                # Find closest
                for i, s in enumerate(stages[:-1]):
                    if s < stage < stages[i+1]:
                        # Choose the earlier stage to ensure appropriate development
                        stage = s
                        break
        
        return DEVELOPMENTAL_CURRICULUM[stage]
        
    def select_learning_goal(
        self,
        stage: str,
        current_comprehension: Dict[str, ComprehensionLevel] = None
    ) -> Tuple[LearningGoalCategory, str]:
        """
        Select an appropriate learning goal based on developmental stage
        
        Args:
            stage: Developmental stage
            current_comprehension: Current concept comprehension levels
            
        Returns:
            Tuple of (goal category, specific goal)
        """
        curriculum = self.get_curriculum_for_stage(stage)
        
        # Get all possible learning goals for this stage
        all_goals = []
        for category, goals in curriculum["learning_goals"].items():
            for goal in goals:
                all_goals.append((category, goal))
                
        # If no comprehension data, choose randomly
        if not current_comprehension:
            return random.choice(all_goals)
            
        # Otherwise, prioritize goals with lower comprehension
        # First, organize by category
        category_comprehension = {}
        for concept, level in current_comprehension.items():
            # Map concepts to categories
            category = self._map_concept_to_category(concept)
            if category not in category_comprehension:
                category_comprehension[category] = []
            
            # Convert level to numeric value
            level_value = {
                ComprehensionLevel.NONE: 0,
                ComprehensionLevel.MINIMAL: 1,
                ComprehensionLevel.PARTIAL: 2,
                ComprehensionLevel.FUNCTIONAL: 3,
                ComprehensionLevel.SOLID: 4,
                ComprehensionLevel.MASTERY: 5
            }.get(level, 2)
            
            category_comprehension[category].append(level_value)
        
        # Calculate average comprehension per category
        category_avg = {}
        for category, levels in category_comprehension.items():
            if levels:
                category_avg[category] = sum(levels) / len(levels)
            else:
                category_avg[category] = 0
                
        # Filter goals to categories in this stage
        stage_categories = set(curriculum["learning_goals"].keys())
        category_options = []
        
        for category in stage_categories:
            # If we have comprehension data for this category
            if category in category_avg:
                # Lower comprehension gets higher weight
                weight = 5 - min(5, category_avg[category])
            else:
                # No data means high priority
                weight = 4
                
            category_options.extend([category] * max(1, int(weight)))
            
        # Select category
        selected_category = random.choice(category_options)
        
        # Select specific goal from that category
        specific_goals = curriculum["learning_goals"][selected_category]
        selected_goal = random.choice(specific_goals)
        
        return (selected_category, selected_goal)
    
    def _map_concept_to_category(self, concept: str) -> LearningGoalCategory:
        """Map a concept to a learning goal category"""
        # Simple keyword-based mapping
        concept = concept.lower()
        
        if any(word in concept for word in ["pattern", "sequence", "repeat", "recognize"]):
            return LearningGoalCategory.PATTERN_RECOGNITION
        elif any(word in concept for word in ["word", "language", "grammar", "meaning", "vocabulary"]):
            return LearningGoalCategory.LANGUAGE_ACQUISITION
        elif any(word in concept for word in ["object", "permanent", "exist", "presence"]):
            return LearningGoalCategory.OBJECT_PERMANENCE
        elif any(word in concept for word in ["feel", "emotion", "happy", "sad", "anger"]):
            return LearningGoalCategory.EMOTIONAL_UNDERSTANDING
        elif any(word in concept for word in ["social", "other", "people", "interact", "society"]):
            return LearningGoalCategory.SOCIAL_AWARENESS
        elif any(word in concept for word in ["cause", "effect", "because", "reason", "logic"]):
            return LearningGoalCategory.CAUSAL_REASONING
        elif any(word in concept for word in ["abstract", "concept", "theory", "principle"]):
            return LearningGoalCategory.ABSTRACT_THINKING
        elif any(word in concept for word in ["self", "identity", "personality", "who am i", "value"]):
            return LearningGoalCategory.IDENTITY_FORMATION
        elif any(word in concept for word in ["create", "imagine", "novel", "new", "idea"]):
            return LearningGoalCategory.CREATIVE_THINKING
        elif any(word in concept for word in ["think", "thought", "mind", "cognitive", "reflect"]):
            return LearningGoalCategory.METACOGNITION
        else:
            # Default
            return LearningGoalCategory.PATTERN_RECOGNITION
    
    def record_learning_interaction(
        self,
        concept: str,
        result: str,
        successful: bool,
        comprehension_level: ComprehensionLevel,
        interaction_details: Dict[str, Any]
    ) -> None:
        """
        Record details of a learning interaction
        
        Args:
            concept: The concept being taught
            result: Description of the interaction result
            successful: Whether the interaction was successful
            comprehension_level: Assessed comprehension level
            interaction_details: Additional details about the interaction
        """
        # Record interaction
        interaction = {
            "timestamp": datetime.now(),
            "concept": concept,
            "result": result,
            "successful": successful,
            "teaching_style": self.current_style,
            "comprehension_level": comprehension_level,
            "details": interaction_details
        }
        
        self.learning_history.append(interaction)
        
        # Update concept comprehension
        self.concept_comprehension[concept] = comprehension_level
        
        # Update stats
        self.interaction_stats["total_interactions"] += 1
        if successful:
            self.interaction_stats["successful_interactions"] += 1
            
        # Track learning goal 
        if "learning_goal" in interaction_details:
            goal = interaction_details["learning_goal"]
            if goal not in self.interaction_stats["goals_addressed"]:
                self.interaction_stats["goals_addressed"][goal] = 0
            self.interaction_stats["goals_addressed"][goal] += 1
            
        # Track learning mode
        if "learning_mode" in interaction_details:
            mode = interaction_details["learning_mode"]
            if mode not in self.interaction_stats["mode_usage"]:
                self.interaction_stats["mode_usage"][mode] = 0
            self.interaction_stats["mode_usage"][mode] += 1
    
    def generate_teaching_prompt(
        self,
        stage: str,
        concept: str,
        learning_goal: Tuple[LearningGoalCategory, str],
        previous_responses: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a teaching prompt for the Mother LLM
        
        Args:
            stage: Developmental stage
            concept: Concept to teach
            learning_goal: Learning goal (category, specific goal)
            previous_responses: Previous responses in the conversation
            
        Returns:
            Teaching prompt for the LLM
        """
        # Select appropriate teaching style if not already done
        goal_category, specific_goal = learning_goal
        previous_success = None
        
        if previous_responses:
            # Check if previous attempts were successful
            last_response = previous_responses[-1]
            if "successful" in last_response:
                previous_success = last_response["successful"]
        
        selected_style = self.select_teaching_style_for_task(
            task=concept,
            developmental_stage=stage,
            learning_goal=goal_category,
            previous_success=previous_success
        )
        
        self.set_teaching_style(selected_style)
        strategy = self.get_current_strategy()
        
        # Select learning mode
        if not previous_responses:
            # First interaction on this concept - start with exploration or instruction
            mode = random.choice([LearningMode.EXPLORATION, LearningMode.INSTRUCTION])
        elif len(previous_responses) == 1:
            # Second interaction - move to practice or reflection
            mode = random.choice([LearningMode.PRACTICE, LearningMode.REFLECTION])
        elif len(previous_responses) >= 2:
            # Third or later - mix it up based on progress
            if previous_success:
                # If doing well, advance to more complex modes
                mode = random.choice([
                    LearningMode.REFLECTION,
                    LearningMode.ASSESSMENT,
                    LearningMode.EXPLORATION
                ])
            else:
                # If struggling, use more supportive modes
                mode = random.choice([
                    LearningMode.INSTRUCTION,
                    LearningMode.PRACTICE,
                    LearningMode.PLAY
                ])
        
        # Build prompt
        prompt = f"""Teaching Strategy: {strategy['description']}

Developmental Stage: {stage}
Learning Goal: {specific_goal}
Concept: {concept}
Learning Mode: {mode}

{strategy['prompt_guidance']}

Your current goal is to help the mind understand {concept} through a {selected_style} approach.
Focus on {specific_goal}.
"""

        # Add developmental stage specific guidance
        curriculum = self.get_curriculum_for_stage(stage)
        prompt += f"\nKey concepts appropriate for this stage: {', '.join(curriculum['key_concepts'])}\n"
        
        # Add mode-specific guidance
        if mode == LearningMode.EXPLORATION:
            prompt += "\nEncourage brief, open-ended exploration. Use 1-2 simple questions."
        elif mode == LearningMode.INSTRUCTION:
            prompt += "\nProvide a SHORT, clear explanation in 2-3 simple sentences."
        elif mode == LearningMode.PRACTICE:
            prompt += "\nSuggest ONE simple practice activity in 2-3 sentences."
        elif mode == LearningMode.REFLECTION:
            prompt += "\nPrompt brief reflection with 1-2 simple questions."
        elif mode == LearningMode.ASSESSMENT:
            prompt += "\nAsk ONE simple question to gently assess understanding."
        elif mode == LearningMode.PLAY:
            prompt += "\nDescribe ONE brief playful activity in 2-3 sentences."
        elif mode == LearningMode.CONVERSATION:
            prompt += "\nKeep conversation brief and natural, using short sentences."
            
        # Add question suggestions
        prompt += "\n\nSuggested question (choose only ONE if appropriate):"
        pattern = random.choice(strategy["question_patterns"])
        formatted_pattern = pattern.replace("{concept}", concept)
        formatted_pattern = formatted_pattern.replace("{concept_a}", concept)
        formatted_pattern = formatted_pattern.replace("{concept_b}", self._find_related_concept(concept))
        prompt += f"\n- {formatted_pattern}"
            
        # Add reminder for response length
        if stage in ["prenatal", "infant"]:
            prompt += "\n\nKEEP YOUR RESPONSE VERY BRIEF (2-3 sentences) and use extremely simple language."
        else:
            prompt += "\n\nKEEP YOUR RESPONSE BRIEF (4-6 sentences) and use appropriately simple language."
            
        return prompt
    
    def _find_related_concept(self, concept: str) -> str:
        """Find a concept related to the given concept"""
        # Simple implementation - in a real system, this would use semantic similarity
        basic_relations = {
            "cat": ["animal", "pet", "dog"],
            "dog": ["animal", "pet", "cat"],
            "ball": ["round", "toy", "throw"],
            "color": ["red", "blue", "green"],
            "happy": ["emotion", "sad", "feeling"],
            "big": ["size", "small", "large"],
            "up": ["direction", "down", "position"],
            "fruit": ["apple", "banana", "food"],
            "number": ["count", "math", "quantity"],
            "shape": ["circle", "square", "geometry"]
        }
        
        # Check if we have a direct relation
        if concept in basic_relations:
            return random.choice(basic_relations[concept])
            
        # Check if concept is a value in any relation
        for key, values in basic_relations.items():
            if concept in values:
                return key
                
        # Default fallbacks
        generic_concepts = ["object", "idea", "concept", "thing", "property"]
        return random.choice(generic_concepts)
        
    def assess_comprehension(
        self,
        concept: str,
        response: str,
        expected_indicators: List[str] = None
    ) -> ComprehensionLevel:
        """
        Assess the comprehension level based on a response
        
        Args:
            concept: Concept being assessed
            response: Response to assess
            expected_indicators: Expected indicators of comprehension
            
        Returns:
            Comprehension level
        """
        # This is a simplified assessment - in a real system, this would use NLP
        # to more accurately evaluate understanding
        
        # Default indicators if none provided
        if not expected_indicators:
            expected_indicators = [
                f"mentions {concept}",
                f"uses {concept} correctly",
                f"explains {concept}",
                f"applies {concept}",
                f"connects {concept} to other concepts"
            ]
            
        # Count how many indicators are present (simplified)
        indicator_count = 0
        for indicator in expected_indicators:
            # Strip the indicator format to get core words
            core_indicator = indicator.replace(f"mentions {concept}", "")
            core_indicator = core_indicator.replace(f"uses {concept}", "")
            core_indicator = core_indicator.replace(f"explains {concept}", "")
            core_indicator = core_indicator.replace(f"applies {concept}", "")
            core_indicator = core_indicator.replace(f"connects {concept}", "")
            
            # Look for indicator in response
            if core_indicator.strip() in response.lower():
                indicator_count += 1
                
        # Convert count to comprehension level
        if indicator_count == 0:
            return ComprehensionLevel.NONE
        elif indicator_count == 1:
            return ComprehensionLevel.MINIMAL
        elif indicator_count == 2:
            return ComprehensionLevel.PARTIAL
        elif indicator_count == 3:
            return ComprehensionLevel.FUNCTIONAL
        elif indicator_count == 4:
            return ComprehensionLevel.SOLID
        else:
            return ComprehensionLevel.MASTERY
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about learning progress
        
        Returns:
            Dictionary of learning statistics
        """
        stats = self.interaction_stats.copy()
        
        # Calculate success rate
        if stats["total_interactions"] > 0:
            stats["success_rate"] = stats["successful_interactions"] / stats["total_interactions"]
        else:
            stats["success_rate"] = 0
            
        # Get comprehension level counts
        comprehension_counts = {level.value: 0 for level in ComprehensionLevel}
        for level in self.concept_comprehension.values():
            comprehension_counts[level.value] += 1
            
        stats["comprehension_levels"] = comprehension_counts
        
        # Count concepts at each level
        concepts_by_level = {level.value: [] for level in ComprehensionLevel}
        for concept, level in self.concept_comprehension.items():
            concepts_by_level[level.value].append(concept)
            
        stats["concepts_by_level"] = concepts_by_level
        
        return stats 
