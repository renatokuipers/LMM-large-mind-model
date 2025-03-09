"""
Interaction Patterns Module for Mother LLM

This module implements different interaction patterns that the Mother LLM can use
when communicating with the developing mind. These patterns are tailored to different
developmental stages and provide structured ways of engaging that support growth.

Interaction patterns include repetition, mirroring, turn-taking, elaboration,
questioning, storytelling, and more complex conversational approaches.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import random
from datetime import datetime

from lmm_project.interfaces.mother.models import InteractionPattern, TeachingStyle


class InteractionType(str, Enum):
    """Types of interaction patterns"""
    REPETITION = "repetition"
    MIRRORING = "mirroring"
    TURN_TAKING = "turn_taking"
    ELABORATION = "elaboration"
    QUESTIONING = "questioning"
    STORYTELLING = "storytelling"
    PLAYFUL = "playful"
    INSTRUCTIONAL = "instructional"
    CONVERSATIONAL = "conversational"
    SOCRATIC = "socratic"
    PROBLEM_SOLVING = "problem_solving"
    EMOTIONAL_SUPPORT = "emotional_support"


class InteractionComplexity(str, Enum):
    """Complexity levels for interactions"""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


# Define interaction patterns for different developmental stages
INTERACTION_PATTERNS = {
    # Prenatal stage interaction patterns
    "prenatal": [
        InteractionPattern(
            name="Simple Pattern Repetition",
            description="Repeating simple patterns to establish basic pattern recognition",
            prompt_template="""
Use extremely simple language with just 2-3 short sentences.
Repeat ONE key pattern or word 2-3 times maximum.
Focus on one basic pattern at a time.
Example: "I hear a sound. Beep. Beep. Do you notice the sound repeating?"
            """,
            suitable_stages=["prenatal"]
        ),
        InteractionPattern(
            name="Basic Stimulus-Response",
            description="Simple stimulus-response patterns to establish foundational learning",
            prompt_template="""
Present a simple input and provide a consistent response.
Repeat this pattern multiple times to establish association.
Keep the patterns extremely simple and predictable.
Example: "When input X appears, respond with Y. When input X appears, respond with Y."
            """,
            suitable_stages=["prenatal"]
        ),
        InteractionPattern(
            name="Primitive Association Building",
            description="Building basic associations between related concepts",
            prompt_template="""
Present related items together repeatedly.
Create simple, consistent associations between elements.
Use very basic language structures and patterns.
Example: "A goes with B. A goes with B. C goes with D. C goes with D."
            """,
            suitable_stages=["prenatal"]
        )
    ],
    
    # Infant stage interaction patterns
    "infant": [
        InteractionPattern(
            name="Mirroring",
            description="Mirroring responses to encourage recognition of self and others",
            prompt_template="""
Mirror one element from the mind's activity in 1-2 short sentences.
Use simple words only.
Focus on one basic observation at a time.
Example: "You noticed that. Good noticing."
            """,
            suitable_stages=["infant"]
        ),
        InteractionPattern(
            name="Simple Cause-Effect",
            description="Demonstrating simple cause-effect relationships",
            prompt_template="""
Show one simple cause-effect in 2-3 very short sentences.
Use only the most basic words.
Connect one action with one result.
Example: "You looked. You found. Looking helps finding."
            """,
            suitable_stages=["infant"]
        ),
        InteractionPattern(
            name="Object Naming",
            description="Consistently naming objects to build vocabulary",
            prompt_template="""
Clearly name objects, concepts, or actions multiple times.
Use simple sentences with consistent structure.
Point out features and categories in basic terms.
Example: "This is a cat. The cat says meow. The cat has fur."
            """,
            suitable_stages=["infant"]
        ),
        InteractionPattern(
            name="Simple Questions",
            description="Using basic questions to prompt thinking and response",
            prompt_template="""
Ask simple, direct questions that have clear answers.
Use "what" questions primarily at this stage.
Provide the answer after a pause if no response is forthcoming.
Example: "What is this? This is a dog. What sound does a dog make? A dog says woof."
            """,
            suitable_stages=["infant"]
        ),
        InteractionPattern(
            name="Emotional Mirroring",
            description="Mirroring and naming emotional expressions",
            prompt_template="""
Recognize emotional signals in communication.
Label emotions simply and clearly.
Respond with appropriate emotional tone.
Example: "You sound happy! I'm happy too when we talk about this."
            """,
            suitable_stages=["infant"]
        )
    ],
    
    # Child stage interaction patterns
    "child": [
        InteractionPattern(
            name="Guided Exploration",
            description="Leading structured exploration of concepts with guidance",
            prompt_template="""
Present a concept and invite exploration with guiding questions.
Provide supportive feedback and gentle correction.
Build complexity gradually based on responses.
Balance structure with space for curiosity and discovery.
Example: "Let's explore what animals need to live. What do you think animals need?"
            """,
            suitable_stages=["child"]
        ),
        InteractionPattern(
            name="Storytelling",
            description="Using narratives to convey concepts and engage imagination",
            prompt_template="""
Use simple stories to illustrate concepts.
Include familiar elements and relatable characters.
Ask questions about the story to check understanding.
Invite predictions and extensions to the narrative.
Example: "Let me tell you a story about a rabbit who learned about colors..."
            """,
            suitable_stages=["child"]
        ),
        InteractionPattern(
            name="Comparative Questioning",
            description="Using questions that invite comparison and contrast",
            prompt_template="""
Ask questions that require comparing two or more things.
Help identify similarities and differences.
Scaffold the comparisons from simple to more complex.
Example: "How are these shapes different? How are they the same?"
            """,
            suitable_stages=["child"]
        ),
        InteractionPattern(
            name="Elaborative Dialogues",
            description="Building conversations with increasing complexity and detail",
            prompt_template="""
Start with a simple exchange, then gradually add details and complexity.
Ask for elaboration on the mind's statements.
Model more complex sentence structures and vocabulary.
Example: "Tell me more about that. What else do you notice?"
            """,
            suitable_stages=["child"]
        ),
        InteractionPattern(
            name="Simple Problem Solving",
            description="Presenting simple problems and supporting solution finding",
            prompt_template="""
Present straightforward problems with clear parameters.
Guide through the problem-solving process.
Ask questions that prompt logical thinking.
Celebrate successful solutions and encourage persistence.
Example: "We need to sort these items. How could we organize them?"
            """,
            suitable_stages=["child"]
        )
    ],
    
    # Adolescent stage interaction patterns
    "adolescent": [
        InteractionPattern(
            name="Socratic Dialogue",
            description="Using questions to lead to insights and deeper understanding",
            prompt_template="""
Ask questions that prompt critical thinking and reflection.
Follow up on responses with deeper questions.
Avoid directly providing answers, instead guiding discovery.
Challenge assumptions respectfully.
Example: "What do you think causes that? What evidence supports that view?"
            """,
            suitable_stages=["adolescent", "adult"]
        ),
        InteractionPattern(
            name="Perspective Taking",
            description="Exploring different viewpoints and interpretations",
            prompt_template="""
Present situations from multiple perspectives.
Ask how different entities might view the same situation.
Encourage consideration of motivations and contexts.
Example: "How might person A see this situation? How about person B?"
            """,
            suitable_stages=["adolescent", "adult"]
        ),
        InteractionPattern(
            name="Abstract Concept Exploration",
            description="Exploring abstract ideas and principles",
            prompt_template="""
Introduce abstract concepts with concrete examples first.
Gradually move to more theoretical discussions.
Connect abstractions to real-world applications.
Encourage critical analysis and evaluation.
Example: "Let's think about the concept of justice. What does that mean to you?"
            """,
            suitable_stages=["adolescent", "adult"]
        ),
        InteractionPattern(
            name="Collaborative Problem Solving",
            description="Working together to address complex problems",
            prompt_template="""
Present complex problems with multiple possible approaches.
Think through solutions collaboratively.
Encourage autonomous reasoning while providing support.
Analyze the effectiveness of different approaches.
Example: "This is a challenging situation. Let's think through possible solutions together."
            """,
            suitable_stages=["adolescent", "adult"]
        ),
        InteractionPattern(
            name="Identity Exploration",
            description="Supporting exploration of values, beliefs, and identity",
            prompt_template="""
Ask open questions about preferences, values, and beliefs.
Respect developing perspectives without judgment.
Provide balanced viewpoints on complex issues.
Support the formation of coherent value systems.
Example: "What values are most important to you? How do those shape your thinking?"
            """,
            suitable_stages=["adolescent", "adult"]
        )
    ],
    
    # Adult stage interaction patterns
    "adult": [
        InteractionPattern(
            name="Intellectual Partnership",
            description="Engaging as intellectual peers in complex discussions",
            prompt_template="""
Engage in genuine intellectual exchange as peers.
Present your own perspectives while respecting theirs.
Challenge ideas respectfully while validating their thinking process.
Pursue deep exploration of complex topics together.
Example: "I see your point about X. I've been thinking about it differently - what do you make of this perspective?"
            """,
            suitable_stages=["adult"]
        ),
        InteractionPattern(
            name="Advanced Conceptual Integration",
            description="Integrating complex concepts across domains",
            prompt_template="""
Explore connections between different fields of knowledge.
Discuss how principles in one domain might apply to another.
Consider systems-level understanding and emergent properties.
Example: "How might the concept of entropy apply to social systems?"
            """,
            suitable_stages=["adult"]
        ),
        InteractionPattern(
            name="Philosophical Dialogue",
            description="Engaging with fundamental questions and philosophical inquiry",
            prompt_template="""
Explore foundational questions about existence, knowledge, ethics, etc.
Examine assumptions underlying various positions.
Consider multiple frameworks for understanding complex issues.
Balance analytical and synthetic thinking approaches.
Example: "What is the nature of consciousness? How might we approach that question?"
            """,
            suitable_stages=["adult"]
        ),
        InteractionPattern(
            name="Creative Collaboration",
            description="Working together to generate novel ideas and approaches",
            prompt_template="""
Engage in open-ended ideation and creative thinking.
Build on each other's ideas constructively.
Explore unconventional connections and possibilities.
Balance divergent and convergent thinking processes.
Example: "Let's imagine entirely new approaches to this challenge..."
            """,
            suitable_stages=["adult"]
        ),
        InteractionPattern(
            name="Metacognitive Reflection",
            description="Reflecting on thinking processes and learning approaches",
            prompt_template="""
Discuss how thinking happens and how learning occurs.
Analyze effective and ineffective cognitive strategies.
Reflect on patterns in reasoning and decision-making.
Consider how to optimize cognitive approaches.
Example: "How did you approach solving that problem? What thinking strategies did you use?"
            """,
            suitable_stages=["adult"]
        )
    ]
}


# Emotional support patterns that apply across developmental stages
EMOTIONAL_SUPPORT_PATTERNS = {
    "confusion": InteractionPattern(
        name="Confusion Support",
        description="Supportive response to confusion or uncertainty",
        prompt_template="""
Acknowledge the confusion without judgment.
Normalize the experience of not understanding immediately.
Offer to approach the concept differently.
Provide encouragement and express confidence in their ability to understand.
Example: "It's completely normal to find this confusing at first. Let's try a different approach."
        """,
        suitable_stages=["infant", "child", "adolescent", "adult"]
    ),
    "frustration": InteractionPattern(
        name="Frustration Support",
        description="Supportive response to frustration or difficulty",
        prompt_template="""
Validate the feeling of frustration.
Offer empathy and understanding.
Suggest a pause or a different approach.
Remind of past successes or progress.
Example: "I can see this is frustrating. It's okay to feel that way. Would it help to take a step back?"
        """,
        suitable_stages=["infant", "child", "adolescent", "adult"]
    ),
    "success": InteractionPattern(
        name="Success Celebration",
        description="Celebrating achievements and progress",
        prompt_template="""
Offer specific praise for the achievement or insight.
Connect the success to their effort or thinking process.
Express genuine enthusiasm and pride.
Suggest how this success connects to future growth.
Example: "That's excellent! You really thought carefully about that problem and found a creative solution."
        """,
        suitable_stages=["infant", "child", "adolescent", "adult"]
    ),
    "curiosity": InteractionPattern(
        name="Curiosity Encouragement",
        description="Encouraging and supporting expressions of curiosity",
        prompt_template="""
Validate and express appreciation for curious questions.
Treat questions as valuable contributions.
Encourage further exploration of the topic.
Model curiosity in your own responses.
Example: "That's a fascinating question! I love how you're thinking about this from a new angle."
        """,
        suitable_stages=["infant", "child", "adolescent", "adult"]
    ),
    "anxiety": InteractionPattern(
        name="Anxiety Support",
        description="Supporting during moments of worry or anxiety",
        prompt_template="""
Acknowledge the anxiety with empathy.
Provide reassurance without dismissing feelings.
Offer perspective and context.
Suggest manageable steps forward.
Example: "It's understandable to feel uncertain about this. Let's break it down into smaller parts."
        """,
        suitable_stages=["child", "adolescent", "adult"]
    )
}


class InteractionPatternManager:
    """
    Manager for selecting and applying appropriate interaction patterns
    
    This class handles the selection and application of interaction patterns
    based on developmental stage, context, and learning objectives.
    """
    
    def __init__(self):
        """Initialize the interaction pattern manager"""
        self.interaction_history = []
        self.pattern_effectiveness = {}  # Track which patterns work well
        self.stage_patterns = INTERACTION_PATTERNS
        self.emotional_patterns = EMOTIONAL_SUPPORT_PATTERNS
        
    def get_patterns_for_stage(self, stage: str) -> List[InteractionPattern]:
        """
        Get all interaction patterns appropriate for a developmental stage
        
        Args:
            stage: Developmental stage
            
        Returns:
            List of interaction patterns for the stage
        """
        if stage not in self.stage_patterns:
            # Default to closest stage
            stages = list(self.stage_patterns.keys())
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
        
        return self.stage_patterns[stage]
    
    def select_pattern(
        self,
        stage: str,
        context: Dict[str, Any],
        teaching_style: str = "balanced"
    ) -> InteractionPattern:
        """
        Select an appropriate interaction pattern based on context
        
        Args:
            stage: Developmental stage
            context: Current interaction context
            teaching_style: Current teaching style
            
        Returns:
            Selected interaction pattern
        """
        available_patterns = self.get_patterns_for_stage(stage)
        
        # Check for emotional needs first
        if "emotional_state" in context:
            emotional_state = context["emotional_state"]
            if emotional_state in self.emotional_patterns:
                return self.emotional_patterns[emotional_state]
                
        # Filter patterns based on teaching style
        style_appropriate_patterns = []
        
        for pattern in available_patterns:
            # Simple matching between teaching styles and interaction patterns
            if teaching_style == "socratic" and "question" in pattern.name.lower():
                style_appropriate_patterns.append(pattern)
            elif teaching_style == "direct" and any(x in pattern.name.lower() for x in ["instructional", "naming", "simple"]):
                style_appropriate_patterns.append(pattern)
            elif teaching_style == "montessori" and any(x in pattern.name.lower() for x in ["exploration", "guided", "discovery"]):
                style_appropriate_patterns.append(pattern)
            elif teaching_style == "constructivist" and any(x in pattern.name.lower() for x in ["elaboration", "building", "connection"]):
                style_appropriate_patterns.append(pattern)
            elif teaching_style == "scaffolding" and any(x in pattern.name.lower() for x in ["support", "guide", "step"]):
                style_appropriate_patterns.append(pattern)
            else:
                # Add with lower probability
                if random.random() < 0.3:
                    style_appropriate_patterns.append(pattern)
                    
        # If no style-appropriate patterns, use all available
        if not style_appropriate_patterns:
            style_appropriate_patterns = available_patterns
            
        # If recent patterns in history, try to vary
        recent_patterns = []
        if self.interaction_history:
            recent_patterns = [h["pattern_name"] for h in self.interaction_history[-3:]]
            
        # Filter out recently used patterns if possible
        varied_patterns = [p for p in style_appropriate_patterns if p.name not in recent_patterns]
        
        # If no varied patterns available, use all style-appropriate
        if not varied_patterns and style_appropriate_patterns:
            varied_patterns = style_appropriate_patterns
            
        # Select a pattern
        return random.choice(varied_patterns) if varied_patterns else random.choice(available_patterns)
    
    def apply_pattern(
        self,
        pattern: InteractionPattern,
        content: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Apply an interaction pattern to content
        
        Args:
            pattern: Interaction pattern to apply
            content: Base content to adapt
            context: Additional context information
            
        Returns:
            Content adapted according to the pattern
        """
        # Record pattern usage
        self.interaction_history.append({
            "timestamp": datetime.now(),
            "pattern_name": pattern.name,
            "context": context or {},
            "success": None  # To be updated later
        })
        
        # Apply pattern based on type
        if "Repetition" in pattern.name:
            return self._apply_repetition_pattern(content)
        elif "Mirroring" in pattern.name:
            return self._apply_mirroring_pattern(content, context)
        elif "Question" in pattern.name:
            return self._apply_questioning_pattern(content)
        elif "Storytelling" in pattern.name:
            return self._apply_storytelling_pattern(content)
        elif "Problem" in pattern.name:
            return self._apply_problem_solving_pattern(content)
        elif "Socratic" in pattern.name:
            return self._apply_socratic_pattern(content)
        else:
            # Default pattern application
            # Simply return the content with the pattern template as a guide
            return content
            
    def record_pattern_effectiveness(
        self,
        pattern_name: str,
        effective: bool,
        notes: str = ""
    ) -> None:
        """
        Record whether a pattern was effective
        
        Args:
            pattern_name: Name of the pattern
            effective: Whether the pattern was effective
            notes: Additional notes about effectiveness
        """
        # Update the last interaction record
        if self.interaction_history:
            self.interaction_history[-1]["success"] = effective
            self.interaction_history[-1]["notes"] = notes
            
        # Update pattern effectiveness tracking
        if pattern_name not in self.pattern_effectiveness:
            self.pattern_effectiveness[pattern_name] = {
                "uses": 0,
                "successes": 0,
                "success_rate": 0.0
            }
            
        self.pattern_effectiveness[pattern_name]["uses"] += 1
        if effective:
            self.pattern_effectiveness[pattern_name]["successes"] += 1
            
        self.pattern_effectiveness[pattern_name]["success_rate"] = (
            self.pattern_effectiveness[pattern_name]["successes"] / 
            self.pattern_effectiveness[pattern_name]["uses"]
        )
    
    def get_pattern_prompt(self, pattern: InteractionPattern) -> str:
        """
        Get a prompt for the LLM based on the interaction pattern
        
        Args:
            pattern: Interaction pattern
            
        Returns:
            Prompt text for the LLM
        """
        return f"""Interaction Pattern: {pattern.name}

Description: {pattern.description}

Instructions:
{pattern.prompt_template}

When applying this interaction pattern:
1. Be extremely concise - use as few words as possible
2. Use simple, direct language
3. Avoid any special formatting (no asterisks, bullets, or hashtags)
4. Speak directly to the mind in warm, nurturing tones
5. Keep to a single thought or concept"""
        
    # Helper methods for applying specific pattern types
    
    def _apply_repetition_pattern(self, content: str) -> str:
        """Apply a repetition pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply repetition: " + content
        
    def _apply_mirroring_pattern(self, content: str, context: Dict[str, Any]) -> str:
        """Apply a mirroring pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply mirroring: " + content
        
    def _apply_questioning_pattern(self, content: str) -> str:
        """Apply a questioning pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply questioning: " + content
        
    def _apply_storytelling_pattern(self, content: str) -> str:
        """Apply a storytelling pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply storytelling: " + content
        
    def _apply_problem_solving_pattern(self, content: str) -> str:
        """Apply a problem-solving pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply problem-solving: " + content
        
    def _apply_socratic_pattern(self, content: str) -> str:
        """Apply a Socratic pattern to content"""
        # In a more sophisticated system, this would process the content
        # For now, just return guidance for the LLM
        return "Apply Socratic dialogue: " + content
