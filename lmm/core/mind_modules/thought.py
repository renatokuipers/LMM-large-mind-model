"""
Thought module for the Large Mind Model (LMM).

This module implements sophisticated thought generation capabilities including:
- Various cognitive processes (reasoning, problem-solving, decision-making)
- Different thought types (analytical, creative, critical, abstract, concrete)
- Developmental stages of thought complexity
- Integration with memory, emotion, and consciousness modules
- Internal dialogue and self-questioning mechanisms
- Attention and distraction modeling
- Thought sequencing and association
- Cognitive biases and heuristics
"""
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
import random
import math
from enum import Enum
from collections import deque, defaultdict
import numpy as np

from pydantic import BaseModel, Field, field_validator
from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.mind_modules.base import MindModule
from lmm.core.development.stages import DevelopmentalStage
from lmm.memory.advanced_memory import MemoryStrength, MemoryActivation

logger = get_logger("lmm.mind_modules.thought")

class ThoughtType(str, Enum):
    """Types of thoughts that can be generated."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    ABSTRACT = "abstract"
    CONCRETE = "concrete"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    METACOGNITIVE = "metacognitive"

class CognitiveProcess(str, Enum):
    """Different cognitive processes involved in thinking."""
    REASONING = "reasoning"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    PATTERN_RECOGNITION = "pattern_recognition"
    ABSTRACTION = "abstraction"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"

class ThoughtContent(BaseModel):
    """Content and metadata of a single thought."""
    id: str = Field(..., description="Unique identifier for the thought")
    content: str = Field(..., description="The actual thought content")
    type: ThoughtType = Field(..., description="Type of thought")
    processes: List[CognitiveProcess] = Field(default_factory=list, description="Cognitive processes involved")
    associations: List[str] = Field(default_factory=list, description="Associated thought IDs")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    complexity: float = Field(0.0, ge=0.0, le=1.0, description="Thought complexity score")
    certainty: float = Field(0.0, ge=0.0, le=1.0, description="Confidence level in the thought")
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('complexity', 'certainty')
    @classmethod
    def validate_float_range(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

class AttentionState(BaseModel):
    """Models the current attention and focus state."""
    focus_level: float = Field(1.0, description="Current focus level (0.0-1.0)")
    distractions: List[str] = Field(default_factory=list, description="Current distracting thoughts")
    attention_capacity: float = Field(1.0, description="Maximum attention capacity")
    sustained_focus_duration: float = Field(0.0, description="Duration of current focus in seconds")

class CognitiveBias(BaseModel):
    """Represents a cognitive bias that can influence thinking."""
    name: str = Field(..., description="Name of the cognitive bias")
    influence_strength: float = Field(..., description="How strongly this bias affects thinking")
    activation_threshold: float = Field(..., description="When this bias becomes active")
    contexts: List[str] = Field(default_factory=list, description="Contexts where this bias applies")

class ThoughtModule(MindModule):
    """
    Implements sophisticated thought generation and processing.
    
    This module handles:
    - Generation of various types of thoughts
    - Management of cognitive processes
    - Integration with other mind modules
    - Development of thinking capabilities
    - Attention and focus management
    - Thought history and reflection
    """
    
    def __init__(self):
        """Initialize the Thought Module."""
        super().__init__("Thought")
        
        # Core thought processing components
        self.current_thoughts = deque(maxlen=10)  # Working memory limit
        self.thought_history = []
        self.thought_associations = defaultdict(set)
        
        # Cognitive process management
        self.active_processes: Set[CognitiveProcess] = set()
        self.process_depths = defaultdict(int)
        
        # Attention and focus management
        self.attention_state = AttentionState()
        self.last_attention_update = datetime.now()
        
        # Development tracking
        self.cognitive_capabilities = {
            "abstraction": 0.0,
            "complexity": 0.0,
            "creativity": 0.0,
            "critical_thinking": 0.0,
            "metacognition": 0.0
        }
        
        # Initialize cognitive biases based on development stage
        self.cognitive_biases = self._initialize_cognitive_biases()
        
        # Integration components
        self.emotional_state = {}
        self.active_memories = set()
        self.consciousness_level = 0.0
        
        logger.info("Initialized Thought Module with advanced cognitive capabilities")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for thought-related operations.
        
        Args:
            input_data: Dictionary containing
                - operation: Operation to perform
                - content: Input content to think about
                - context: Current context
                - emotional_state: Current emotional state
                - consciousness_state: Current consciousness state
                - memory_activations: Active memories
                - developmental_stage: Current developmental stage
                
        Returns:
            Dictionary with operation results
        """
        operation = input_data.get("operation", "generate_thought")
        stage = input_data.get("developmental_stage", DevelopmentalStage.PRENATAL.value)
        
        # Update developmental parameters
        self._update_developmental_parameters(stage)
        
        # Update integration states
        self._update_integration_states(input_data)
        
        # Process operation
        results = {"success": False, "operation": operation}
        
        try:
            if operation == "generate_thought":
                results = self._generate_thought(input_data)
            elif operation == "analyze_thought":
                results = self._analyze_thought(input_data)
            elif operation == "associate_thoughts":
                results = self._associate_thoughts(input_data)
            elif operation == "reflect":
                results = self._reflect_on_thoughts(input_data)
            elif operation == "focus":
                results = self._manage_attention(input_data)
            else:
                results["error"] = f"Unknown operation: {operation}"
            
            # Update thought history
            if results.get("thought"):
                self._update_thought_history(results["thought"])
            
        except Exception as e:
            logger.error(f"Error in thought processing: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _generate_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new thought based on input and current state."""
        content = input_data.get("content", "")
        context = input_data.get("context", {})
        
        # Apply attention filter
        if not self._check_attention_capacity():
            return {
                "success": False,
                "error": "Insufficient attention capacity",
                "current_attention": self.attention_state.focus_level
            }
        
        # Determine thought type and processes based on content and context
        thought_type = self._determine_thought_type(content, context)
        processes = self._determine_cognitive_processes(thought_type, content)
        
        # Apply cognitive biases
        content = self._apply_cognitive_biases(content, context)
        
        # Generate thought with appropriate complexity for developmental stage
        complexity = self._calculate_thought_complexity(content, processes)
        certainty = self._calculate_thought_certainty(complexity, processes)
        
        thought = ThoughtContent(
            id=f"thought_{datetime.now().timestamp()}",
            content=content,
            type=thought_type,
            processes=processes,
            context=context,
            complexity=complexity,
            certainty=certainty
        )
        
        # Add to current thoughts and update associations
        self._update_current_thoughts(thought)
        
        return {
            "success": True,
            "thought": thought.model_dump(),
            "attention_state": self.attention_state.model_dump()
        }
    
    def _analyze_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a thought for patterns, implications, and connections."""
        thought_id = input_data.get("thought_id")
        if not thought_id:
            return {"success": False, "error": "No thought ID provided"}
        
        # Find thought in history
        thought = next((t for t in self.thought_history if t.id == thought_id), None)
        if not thought:
            return {"success": False, "error": "Thought not found"}
        
        # Analyze patterns and implications
        patterns = self._identify_patterns(thought)
        implications = self._derive_implications(thought)
        connections = self._find_connections(thought)
        
        return {
            "success": True,
            "thought_id": thought_id,
            "patterns": patterns,
            "implications": implications,
            "connections": connections
        }
    
    def _associate_thoughts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create associations between thoughts."""
        thought_ids = input_data.get("thought_ids", [])
        if len(thought_ids) < 2:
            return {"success": False, "error": "Need at least two thoughts to associate"}
        
        # Create bidirectional associations
        for i, thought_id in enumerate(thought_ids):
            for other_id in thought_ids[i+1:]:
                self.thought_associations[thought_id].add(other_id)
                self.thought_associations[other_id].add(thought_id)
        
        return {
            "success": True,
            "associations_created": len(thought_ids) * (len(thought_ids) - 1) // 2
        }
    
    def _reflect_on_thoughts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on recent thoughts and generate insights."""
        # Get recent thoughts to reflect on
        recent_thoughts = list(self.current_thoughts)
        if not recent_thoughts:
            return {"success": False, "error": "No recent thoughts to reflect on"}
        
        # Analyze patterns and generate insights
        patterns = []
        insights = []
        for thought in recent_thoughts:
            patterns.extend(self._identify_patterns(thought))
            implications = self._derive_implications(thought)
            insights.extend(implications)
        
        # Generate meta-thoughts about the patterns and insights
        meta_thoughts = self._generate_meta_thoughts(patterns, insights)
        
        return {
            "success": True,
            "patterns": patterns,
            "insights": insights,
            "meta_thoughts": meta_thoughts
        }
    
    def _manage_attention(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage attention and focus state."""
        # Update attention state
        current_time = datetime.now()
        time_diff = (current_time - self.last_attention_update).total_seconds()
        
        # Natural focus decay over time
        self.attention_state.focus_level *= math.exp(-0.1 * time_diff)
        
        # Process distractions
        distractions = input_data.get("distractions", [])
        for distraction in distractions:
            self._handle_distraction(distraction)
        
        # Update focus duration
        if self.attention_state.focus_level > 0.7:
            self.attention_state.sustained_focus_duration += time_diff
        else:
            self.attention_state.sustained_focus_duration = 0
        
        self.last_attention_update = current_time
        
        return {
            "success": True,
            "attention_state": self.attention_state.model_dump()
        }
    
    def _update_developmental_parameters(self, stage: str) -> None:
        """Update cognitive capabilities based on developmental stage."""
        stage_capabilities = {
            DevelopmentalStage.PRENATAL.value: {
                "abstraction": 0.1,
                "complexity": 0.1,
                "creativity": 0.2,
                "critical_thinking": 0.1,
                "metacognition": 0.0
            },
            DevelopmentalStage.INFANCY.value: {
                "abstraction": 0.2,
                "complexity": 0.2,
                "creativity": 0.3,
                "critical_thinking": 0.2,
                "metacognition": 0.1
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "abstraction": 0.4,
                "complexity": 0.4,
                "creativity": 0.5,
                "critical_thinking": 0.3,
                "metacognition": 0.3
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "abstraction": 0.6,
                "complexity": 0.6,
                "creativity": 0.7,
                "critical_thinking": 0.5,
                "metacognition": 0.5
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "abstraction": 0.8,
                "complexity": 0.8,
                "creativity": 0.8,
                "critical_thinking": 0.7,
                "metacognition": 0.7
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "abstraction": 1.0,
                "complexity": 1.0,
                "creativity": 1.0,
                "critical_thinking": 1.0,
                "metacognition": 1.0
            }
        }
        
        self.cognitive_capabilities = stage_capabilities.get(
            stage,
            stage_capabilities[DevelopmentalStage.PRENATAL.value]
        )
    
    def _initialize_cognitive_biases(self) -> List[CognitiveBias]:
        """Initialize cognitive biases with developmental considerations."""
        return [
            CognitiveBias(
                name="confirmation_bias",
                influence_strength=0.7,
                activation_threshold=0.3,
                contexts=["decision_making", "belief_formation"]
            ),
            CognitiveBias(
                name="availability_heuristic",
                influence_strength=0.6,
                activation_threshold=0.2,
                contexts=["risk_assessment", "probability_estimation"]
            ),
            CognitiveBias(
                name="anchoring_bias",
                influence_strength=0.5,
                activation_threshold=0.4,
                contexts=["numerical_estimation", "value_assessment"]
            ),
            # Add more biases as needed
        ]
    
    def _update_integration_states(self, input_data: Dict[str, Any]) -> None:
        """Update states from other modules for integration."""
        self.emotional_state = input_data.get("emotional_state", {})
        self.active_memories = set(input_data.get("memory_activations", []))
        self.consciousness_level = input_data.get("consciousness_state", {}).get("level", 0.0)
    
    def _determine_thought_type(self, content: str, context: Dict[str, Any]) -> ThoughtType:
        """
        Determine the most appropriate thought type based on content and context.
        
        This uses natural language patterns, context cues, and developmental stage
        to identify the most likely thought type.
        
        Args:
            content: The thought content text
            context: Contextual information
            
        Returns:
            The identified thought type
        """
        # Get development level to calibrate sophistication
        abstraction_level = self.cognitive_capabilities["abstraction"]
        creativity_level = self.cognitive_capabilities["creativity"]
        critical_level = self.cognitive_capabilities["critical_thinking"]
        
        # Prepare content for analysis
        content_lower = content.lower()
        
        # Check context for explicit thought type request
        if context.get("requested_thought_type"):
            requested_type = context["requested_thought_type"]
            try:
                return ThoughtType(requested_type)
            except ValueError:
                logger.warning(f"Invalid requested thought type: {requested_type}")
        
        # Define pattern markers for each thought type
        type_indicators = {
            ThoughtType.ANALYTICAL: [
                "analyze", "examine", "compare", "what if", "factor", "logic", "reason",
                "therefore", "cause", "effect", "relationship", "calculate", "evaluate"
            ],
            ThoughtType.CREATIVE: [
                "imagine", "create", "design", "invent", "novel", "unique", "different",
                "could be", "might be", "visualize", "dream", "inspiration", "possibility"
            ],
            ThoughtType.CRITICAL: [
                "critique", "problem", "flaw", "mistake", "error", "wrong", "better",
                "improve", "should be", "however", "nevertheless", "evaluate", "question"
            ],
            ThoughtType.ABSTRACT: [
                "concept", "theory", "principle", "generally", "abstract", "philosophical",
                "meaning", "represent", "symbolize", "essence", "fundamental", "universal"
            ],
            ThoughtType.CONCRETE: [
                "specifically", "example", "instance", "case", "practical", "tangible",
                "real", "physical", "specific", "particular", "detailed", "exact", "precisely"
            ],
            ThoughtType.EMOTIONAL: [
                "feel", "emotion", "happy", "sad", "angry", "afraid", "joy", "love", "hate",
                "worried", "excited", "nervous", "mood", "attitude", "heart", "passion"
            ],
            ThoughtType.SOCIAL: [
                "people", "person", "friend", "family", "relationship", "together", "community",
                "group", "society", "interaction", "communicate", "share", "understand"
            ],
            ThoughtType.METACOGNITIVE: [
                "thinking about", "reflect", "aware", "conscious", "metacognition", "my thought",
                "my mind", "my understanding", "know that I", "realize that I", "notice that I"
            ]
        }
        
        # Score each thought type based on indicators in content
        type_scores = {thought_type: 0.0 for thought_type in ThoughtType}
        
        for thought_type, indicators in type_indicators.items():
            # Count occurrences of indicators
            indicator_count = sum(1 for indicator in indicators if indicator in content_lower)
            # Weight by indicator matches and list size
            type_scores[thought_type] = indicator_count / max(1, len(indicators))
        
        # Apply cognitive capability modifiers
        # Higher abstraction enables more abstract thought
        type_scores[ThoughtType.ABSTRACT] *= (0.2 + 0.8 * abstraction_level)
        # Higher creativity enables more creative thought
        type_scores[ThoughtType.CREATIVE] *= (0.3 + 0.7 * creativity_level)
        # Higher critical thinking enables more critical thought
        type_scores[ThoughtType.CRITICAL] *= (0.3 + 0.7 * critical_level)
        # Metacognition requires sufficient development
        type_scores[ThoughtType.METACOGNITIVE] *= (0.1 + 0.9 * self.cognitive_capabilities["metacognition"])
        
        # Consider emotional state for emotional thought boost
        if self.emotional_state:
            emotion_intensity = sum(self.emotional_state.values()) / len(self.emotional_state)
            type_scores[ThoughtType.EMOTIONAL] += 0.3 * emotion_intensity
        
        # Identify top type (with random for ties)
        if all(score == 0 for score in type_scores.values()):
            # Default to concrete for very early development
            if abstraction_level < 0.3:
                return ThoughtType.CONCRETE
            # Default to analytical for typical cases with no clear indicators
            return ThoughtType.ANALYTICAL
        
        # Get the thought type with the highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _determine_cognitive_processes(self, thought_type: ThoughtType, content: str) -> List[CognitiveProcess]:
        """
        Determine which cognitive processes are involved in a thought.
        
        This analyzes the thought type and content to identify which cognitive
        processes are being employed, considering developmental capabilities.
        
        Args:
            thought_type: Type of thought
            content: Thought content
            
        Returns:
            List of cognitive processes involved
        """
        # Track matched processes
        involved_processes = set()
        content_lower = content.lower()
        
        # Process mapping to thought types (primary associations)
        type_to_processes = {
            ThoughtType.ANALYTICAL: {
                CognitiveProcess.REASONING, 
                CognitiveProcess.EVALUATION
            },
            ThoughtType.CREATIVE: {
                CognitiveProcess.SYNTHESIS, 
                CognitiveProcess.ABSTRACTION
            },
            ThoughtType.CRITICAL: {
                CognitiveProcess.EVALUATION, 
                CognitiveProcess.REFLECTION
            },
            ThoughtType.ABSTRACT: {
                CognitiveProcess.ABSTRACTION, 
                CognitiveProcess.REASONING
            },
            ThoughtType.CONCRETE: {
                CognitiveProcess.PATTERN_RECOGNITION
            },
            ThoughtType.EMOTIONAL: {
                CognitiveProcess.REFLECTION
            },
            ThoughtType.SOCIAL: {
                CognitiveProcess.PATTERN_RECOGNITION, 
                CognitiveProcess.REFLECTION
            },
            ThoughtType.METACOGNITIVE: {
                CognitiveProcess.REFLECTION, 
                CognitiveProcess.EVALUATION
            }
        }
        
        # Add primary processes for the thought type
        involved_processes.update(type_to_processes.get(thought_type, set()))
        
        # Process indicator patterns
        process_indicators = {
            CognitiveProcess.REASONING: [
                "because", "therefore", "since", "so", "thus", "consequently",
                "if", "then", "would", "could", "cause", "effect", "reason"
            ],
            CognitiveProcess.PROBLEM_SOLVING: [
                "problem", "solution", "solve", "resolve", "address", "approach",
                "method", "strategy", "tackle", "overcome", "challenge", "fix"
            ],
            CognitiveProcess.DECISION_MAKING: [
                "decide", "choice", "choose", "option", "alternative", "select",
                "best", "worst", "better", "prefer", "decision", "consider"
            ],
            CognitiveProcess.PATTERN_RECOGNITION: [
                "pattern", "similar", "same", "different", "common", "trend",
                "recurring", "recognize", "identify", "detect", "observe"
            ],
            CognitiveProcess.ABSTRACTION: [
                "abstract", "concept", "general", "universal", "principle",
                "theory", "framework", "model", "represent", "symbolize"
            ],
            CognitiveProcess.SYNTHESIS: [
                "combine", "integrate", "merge", "blend", "incorporate",
                "together", "connection", "relationship", "link", "network"
            ],
            CognitiveProcess.EVALUATION: [
                "evaluate", "assess", "judge", "rate", "rank", "compare",
                "value", "worth", "quality", "effectiveness", "efficiency"
            ],
            CognitiveProcess.REFLECTION: [
                "reflect", "think about", "consider", "contemplate", "ponder",
                "introspect", "examine", "review", "reconsider", "revisit"
            ]
        }
        
        # Find secondary processes from content indicators
        for process, indicators in process_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                involved_processes.add(process)
        
        # Limit processes by developmental capability
        final_processes = []
        
        # Establish developmental gating for each process
        process_requirements = {
            CognitiveProcess.REASONING: {"abstraction": 0.2},
            CognitiveProcess.PROBLEM_SOLVING: {"complexity": 0.3},
            CognitiveProcess.DECISION_MAKING: {"complexity": 0.3, "critical_thinking": 0.2},
            CognitiveProcess.PATTERN_RECOGNITION: {"complexity": 0.1},  # Available early
            CognitiveProcess.ABSTRACTION: {"abstraction": 0.5},
            CognitiveProcess.SYNTHESIS: {"creativity": 0.4, "complexity": 0.4},
            CognitiveProcess.EVALUATION: {"critical_thinking": 0.4},
            CognitiveProcess.REFLECTION: {"metacognition": 0.3}
        }
        
        # Filter processes based on developmental capabilities
        for process in involved_processes:
            requirements = process_requirements.get(process, {})
            
            # Check if all requirements are met
            meets_requirements = True
            for capability, min_level in requirements.items():
                if self.cognitive_capabilities.get(capability, 0) < min_level:
                    meets_requirements = False
                    break
            
            if meets_requirements:
                final_processes.append(process)
        
        # Default to pattern recognition if no processes identified (most basic)
        if not final_processes:
            return [CognitiveProcess.PATTERN_RECOGNITION]
        
        return final_processes
    
    def _calculate_thought_complexity(self, content: str, processes: List[CognitiveProcess]) -> float:
        """
        Calculate the complexity of a thought based on linguistic features and cognitive processes.
        
        Args:
            content: Thought content
            processes: Cognitive processes involved
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Get developmental capability cap
        max_complexity = self.cognitive_capabilities["complexity"]
        
        # Base complexity from linguistic features
        linguistic_complexity = self._calculate_linguistic_complexity(content)
        
        # Process-based complexity
        process_weights = {
            CognitiveProcess.PATTERN_RECOGNITION: 0.2,  # Simpler
            CognitiveProcess.REASONING: 0.5,
            CognitiveProcess.PROBLEM_SOLVING: 0.6,
            CognitiveProcess.DECISION_MAKING: 0.5,
            CognitiveProcess.ABSTRACTION: 0.8,  # More complex
            CognitiveProcess.SYNTHESIS: 0.7,
            CognitiveProcess.EVALUATION: 0.6,
            CognitiveProcess.REFLECTION: 0.7
        }
        
        # Calculate average process complexity
        process_complexity = sum(process_weights.get(p, 0.5) for p in processes) / max(1, len(processes))
        
        # Combine linguistic and process complexity
        combined_complexity = 0.6 * linguistic_complexity + 0.4 * process_complexity
        
        # Apply developmental cap (complexity can't exceed capability)
        capped_complexity = min(combined_complexity, max_complexity)
        
        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, capped_complexity))

    def _calculate_linguistic_complexity(self, content: str) -> float:
        """Calculate linguistic complexity based on text features."""
        if not content:
            return 0.0
        
        # Split into sentences and words
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = [w.strip() for w in content.split() if w.strip()]
        
        if not sentences or not words:
            return 0.0
        
        # Simple complexity metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Vocabulary richness (approximation)
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / len(words)
        
        # Calculate various components of complexity
        length_component = min(1.0, avg_sentence_length / 25.0)  # Cap at 25 words/sentence
        word_length_component = min(1.0, avg_word_length / 8.0)  # Cap at 8 letters/word
        diversity_component = min(1.0, lexical_diversity)
        
        # Look for complex structures
        has_conjunctions = any(conj in content.lower() for conj in 
                            ["and", "but", "or", "because", "however", "although", 
                            "therefore", "nevertheless", "despite"])
        has_conditionals = any(cond in content.lower() for cond in 
                            ["if", "when", "unless", "until", "while"])
        
        structure_component = 0.3 * has_conjunctions + 0.4 * has_conditionals
        
        # Combine with weights
        complexity = (
            0.3 * length_component +
            0.2 * word_length_component +
            0.3 * diversity_component +
            0.2 * structure_component
        )
        
        return complexity
    
    def _calculate_thought_certainty(self, complexity: float, processes: List[CognitiveProcess]) -> float:
        """
        Calculate the certainty level of a thought based on complexity and processes.
        
        Args:
            complexity: Thought complexity
            processes: Cognitive processes involved
            
        Returns:
            Certainty score (0.0-1.0)
        """
        # Base certainty inversely related to complexity
        # More complex thoughts tend to have lower certainty
        base_certainty = 1.0 - (0.3 * complexity)
        
        # Process-based certainty modifiers
        process_certainty_modifiers = {
            CognitiveProcess.PATTERN_RECOGNITION: 0.2,  # Pattern recognition increases certainty
            CognitiveProcess.REASONING: 0.1,  # Reasoning slightly increases certainty
            CognitiveProcess.PROBLEM_SOLVING: -0.1,  # Problem solving involves uncertainty
            CognitiveProcess.DECISION_MAKING: -0.1,  # Decision making involves weighing options
            CognitiveProcess.ABSTRACTION: -0.2,  # Abstraction reduces certainty
            CognitiveProcess.SYNTHESIS: -0.1,  # Synthesis has some uncertainty
            CognitiveProcess.EVALUATION: 0.1,  # Evaluation slightly increases certainty
            CognitiveProcess.REFLECTION: -0.2   # Reflection often considers alternatives
        }
        
        # Apply process modifiers
        process_modifier = sum(process_certainty_modifiers.get(p, 0) for p in processes) / max(1, len(processes))
        
        # Critical thinking capability decreases certainty (more nuanced view)
        critical_thinking_effect = -0.2 * self.cognitive_capabilities["critical_thinking"]
        
        # Final certainty calculation
        certainty = base_certainty + process_modifier + critical_thinking_effect
        
        # For very early developmental stages, certainty is higher (black and white thinking)
        if self.cognitive_capabilities["abstraction"] < 0.3:
            certainty += 0.3
        
        # Metacognition decreases certainty (awareness of knowledge limitations)
        metacognition_effect = -0.3 * self.cognitive_capabilities["metacognition"]
        certainty += metacognition_effect
        
        # Emotional state can affect certainty
        if self.emotional_state:
            # High emotional intensity can increase certainty
            primary_emotion = max(self.emotional_state.items(), key=lambda x: x[1], default=(None, 0))
            if primary_emotion[0] and primary_emotion[1] > 0.7:
                certainty += 0.1
        
        # Normalize to 0.0-1.0 range
        return max(0.1, min(1.0, certainty))  # Minimum 0.1 certainty
    
    def _apply_cognitive_biases(self, content: str, context: Dict[str, Any]) -> str:
        """
        Apply relevant cognitive biases to thought content.
        
        This simulates how cognitive biases influence thinking, with bias strength
        varying by developmental stage and context.
        
        Args:
            content: Original thought content
            context: Contextual information
            
        Returns:
            Modified thought content with bias effects
        """
        if not self.cognitive_biases:
            return content
        
        # Create a modified copy of the content
        modified_content = content
        
        # Identify active contexts
        active_contexts = set(context.get("contexts", []))
        
        # Process each cognitive bias
        for bias in self.cognitive_biases:
            # Check if bias is active in current context
            context_active = not bias.contexts or any(c in active_contexts for c in bias.contexts)
            
            # Check if bias activation threshold is reached by random chance
            activation_roll = random.random()
            
            if context_active and activation_roll <= bias.activation_threshold:
                # Apply bias effects based on type
                if bias.name == "confirmation_bias":
                    modified_content = self._apply_confirmation_bias(modified_content, context, bias.influence_strength)
                elif bias.name == "availability_heuristic":
                    modified_content = self._apply_availability_bias(modified_content, context, bias.influence_strength)
                elif bias.name == "anchoring_bias":
                    modified_content = self._apply_anchoring_bias(modified_content, context, bias.influence_strength)
                # Add more bias implementations as needed
        
        return modified_content

    def _apply_confirmation_bias(self, content: str, context: Dict[str, Any], strength: float) -> str:
        """Apply confirmation bias effects to content."""
        # Extract existing beliefs from context
        beliefs = context.get("beliefs", {})
        if not beliefs:
            return content
        
        # Find the strongest belief
        strongest_belief = max(beliefs.items(), key=lambda x: x[1], default=(None, 0))
        if not strongest_belief[0]:
            return content
        
        # Add confirmatory language based on belief strength and bias strength
        confirmation_phrases = [
            f"This confirms my understanding about {strongest_belief[0]}.",
            f"This aligns with what I already know about {strongest_belief[0]}.",
            f"This is consistent with my existing knowledge about {strongest_belief[0]}.",
            f"This makes sense given what I know about {strongest_belief[0]}."
        ]
        
        # Only add confirmation if belief and bias are strong enough
        if strongest_belief[1] * strength > 0.5 and random.random() < strength:
            chosen_phrase = random.choice(confirmation_phrases)
            
            # For more developed cognition, make the bias more subtle
            if self.cognitive_capabilities["metacognition"] > 0.6:
                # More nuanced expressions of confirmation bias
                subtle_phrases = [
                    f"I think this supports the view that {strongest_belief[0]}.",
                    f"This seems to provide evidence for {strongest_belief[0]}.",
                    f"I notice this relates to my understanding of {strongest_belief[0]}."
                ]
                chosen_phrase = random.choice(subtle_phrases)
            
            # Add the bias phrase
            return f"{content} {chosen_phrase}"
        
        return content

    def _apply_availability_bias(self, content: str, context: Dict[str, Any], strength: float) -> str:
        """Apply availability heuristic effects to content."""
        # Get recently accessed memories
        recent_memories = context.get("recent_memories", [])
        if not recent_memories:
            return content
        
        # Choose a recent memory to overweight in the thinking
        memory = random.choice(recent_memories[:3])  # Focus on very recent ones
        
        # Availability bias phrases
        availability_phrases = [
            f"This reminds me of a recent experience with {memory}.",
            f"Like what happened with {memory}, this seems important.",
            f"Based on my recent experience with {memory}, this seems likely.",
            f"The {memory} situation makes me think this is common."
        ]
        
        # Apply bias with probability based on strength
        if random.random() < strength:
            return f"{content} {random.choice(availability_phrases)}"
        
        return content

    def _apply_anchoring_bias(self, content: str, context: Dict[str, Any], strength: float) -> str:
        """Apply anchoring bias effects to content."""
        # Look for anchoring values
        anchor_value = context.get("initial_value") or context.get("first_mentioned_number")
        if not anchor_value:
            return content
        
        # Anchoring bias phrases 
        anchoring_phrases = [
            f"Starting from {anchor_value}, this seems reasonable.",
            f"Considering the initial {anchor_value}, I think this makes sense.",
            f"With {anchor_value} as a reference point, this is my conclusion.",
            f"Based on the {anchor_value} figure mentioned earlier, this follows."
        ]
        
        # Apply bias with probability based on strength
        if random.random() < strength:
            return f"{content} {random.choice(anchoring_phrases)}"
        
        return content
    
    def _check_attention_capacity(self) -> bool:
        """Check if there's sufficient attention capacity for a new thought."""
        return self.attention_state.focus_level > 0.3
    
    def _update_current_thoughts(self, thought: ThoughtContent) -> None:
        """Update the current thoughts queue."""
        self.current_thoughts.append(thought)
        self.thought_history.append(thought)
    
    def _update_thought_history(self, thought: Dict[str, Any]) -> None:
        """Update thought history with new thought."""
        # Implementation would maintain history and prune old thoughts
        pass
    
    def _identify_patterns(self, thought: ThoughtContent) -> List[str]:
        """
        Identify patterns in thought content and between thoughts.
        
        This looks for repeating elements, common themes, and structural patterns
        in the current thought and across recent thoughts.
        
        Args:
            thought: The thought to analyze for patterns
            
        Returns:
            List of identified patterns as descriptive strings
        """
        patterns = []
        
        # Pattern detection depends on metacognitive ability
        pattern_detection_ability = self.cognitive_capabilities["metacognition"]
        if pattern_detection_ability < 0.2:
            return []  # Very limited pattern recognition at early stages
        
        # Get recent thoughts for comparison
        recent_thoughts = list(self.current_thoughts)
        
        # 1. Look for recurring phrases in the current thought
        content = thought.content
        words = [w.lower() for w in content.split() if len(w) > 3]  # Focus on significant words
        
        # Find repeated words
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Identify repeated words as patterns
        repeated_words = [word for word, count in word_counts.items() if count > 1]
        if repeated_words and random.random() < pattern_detection_ability:
            patterns.append(f"Repeated use of: {', '.join(repeated_words[:3])}")
        
        # 2. Compare with recent thoughts to find similarities
        if recent_thoughts and len(recent_thoughts) > 1:
            # Extract thought content from recent thoughts
            recent_contents = [t.content for t in recent_thoughts if t.id != thought.id]
            
            # Compare current thought with previous thoughts
            for prev_content in recent_contents:
                prev_words = set(w.lower() for w in prev_content.split() if len(w) > 3)
                current_words = set(words)
                
                # Find common significant words
                common_words = current_words.intersection(prev_words)
                if len(common_words) >= 3 and random.random() < pattern_detection_ability:
                    patterns.append(f"Theme continuation with words: {', '.join(list(common_words)[:3])}")
                    break  # Just note one thematic connection
        
        # 3. Identify thought type patterns
        if recent_thoughts and len(recent_thoughts) > 2:
            # Count thought types
            type_counts = {}
            for t in recent_thoughts:
                type_counts[t.type] = type_counts.get(t.type, 0) + 1
            
            # Check if current thought continues a pattern
            if type_counts.get(thought.type, 0) > 1:
                patterns.append(f"Continued pattern of {thought.type.value} thinking")
        
        # 4. More sophisticated patterns for higher metacognition
        if pattern_detection_ability > 0.6:
            # Check for cause-effect patterns
            if "because" in content.lower() or "therefore" in content.lower():
                patterns.append("Causal reasoning pattern")
                
            # Check for comparison patterns
            if any(term in content.lower() for term in ["like", "unlike", "similar", "different", "compare"]):
                patterns.append("Comparative reasoning pattern")
                
            # Check for hypothesis forming
            if any(term in content.lower() for term in ["if", "might", "could", "possibly", "perhaps"]):
                patterns.append("Hypothesis formation pattern")
        
        # Limit patterns to capability level
        pattern_capacity = int(2 + 3 * pattern_detection_ability)
        return patterns[:pattern_capacity]
    
    def _derive_implications(self, thought: ThoughtContent) -> List[str]:
        """
        Derive potential implications from a thought.
        
        This analyzes the thought to identify potential consequences,
        future impacts, and logical next steps that follow from it.
        
        Args:
            thought: The thought to analyze for implications
            
        Returns:
            List of implications as descriptive strings
        """
        implications = []
        content = thought.content
        
        # Implication derivation depends on both abstract and critical thinking
        implication_ability = (
            0.7 * self.cognitive_capabilities["abstraction"] +
            0.3 * self.cognitive_capabilities["critical_thinking"]
        )
        
        if implication_ability < 0.3:
            return []  # Limited ability to derive implications
        
        # 1. Direct implications based on thought type
        type_implications = {
            ThoughtType.ANALYTICAL: ["May require further analysis", "Could lead to deeper understanding"],
            ThoughtType.CREATIVE: ["Could inspire new ideas", "Might lead to novel solutions"],
            ThoughtType.CRITICAL: ["May identify problems to solve", "Could improve existing approaches"],
            ThoughtType.ABSTRACT: ["May apply across multiple contexts", "Could reveal underlying principles"],
            ThoughtType.CONCRETE: ["Might apply to specific situations", "Could be directly acted upon"],
            ThoughtType.EMOTIONAL: ["May influence emotional state", "Could affect motivation and engagement"],
            ThoughtType.SOCIAL: ["Might impact interactions with others", "Could enhance social understanding"],
            ThoughtType.METACOGNITIVE: ["May improve thinking processes", "Could lead to better self-awareness"]
        }
        
        # Add type-based implications with probability based on capability
        type_imps = type_implications.get(thought.type, [])
        for imp in type_imps:
            if random.random() < implication_ability:
                implications.append(imp)
        
        # 2. Logical consequence implications
        if CognitiveProcess.REASONING in thought.processes:
            # Look for conditional statements (if-then patterns)
            if "if" in content.lower() and any(term in content.lower() for term in ["then", "would", "could", "might"]):
                implications.append("Suggests conditional outcomes based on premises")
            
            # Look for causal relationships
            if any(term in content.lower() for term in ["because", "cause", "effect", "result", "impact"]):
                implications.append("Indicates causal relationships that may extend to other contexts")
        
        # 3. Knowledge-building implications
        if CognitiveProcess.ABSTRACTION in thought.processes or CognitiveProcess.SYNTHESIS in thought.processes:
            implications.append("May contribute to building broader conceptual understanding")
        
        # 4. Action-oriented implications
        if CognitiveProcess.PROBLEM_SOLVING in thought.processes or CognitiveProcess.DECISION_MAKING in thought.processes:
            implications.append("Could lead to specific actions or decisions")
        
        # 5. More sophisticated implications for higher capability
        if implication_ability > 0.7:
            # Check for far-reaching implications
            if thought.complexity > 0.6:
                implications.append("May have system-wide implications beyond the immediate context")
            
            # Check for counterintuitive implications
            if thought.certainty < 0.5 and CognitiveProcess.EVALUATION in thought.processes:
                implications.append("Could lead to counterintuitive conclusions requiring validation")
                
            # Check for paradigm-shifting implications
            if thought.complexity > 0.8 and thought.certainty > 0.7:
                implications.append("May require fundamental revision of existing understanding")
        
        # Randomly select implications based on capability
        implication_count = int(1 + 4 * implication_ability)
        if len(implications) > implication_count:
            implications = random.sample(implications, implication_count)
        
        return implications
    
    def _find_connections(self, thought: ThoughtContent) -> List[str]:
        """
        Find connections between the given thought and other thoughts.
        
        This identifies semantic, contextual, and structural connections
        between the current thought and previous thoughts.
        
        Args:
            thought: The thought to find connections for
            
        Returns:
            List of connection descriptions
        """
        connections = []
        
        # Connection finding depends on abstraction and creativity
        connection_ability = (
            0.6 * self.cognitive_capabilities["abstraction"] +
            0.4 * self.cognitive_capabilities["creativity"]
        )
        
        if connection_ability < 0.2 or not self.thought_history:
            return []  # Limited ability to find connections
        
        # Current thought details
        content = thought.content.lower()
        content_words = set(w for w in content.split() if len(w) > 3)
        
        # 1. Look for existing associations in the graph
        thought_id = thought.id
        direct_associations = list(self.thought_associations.get(thought_id, set()))
        
        if direct_associations:
            # Describe direct associations
            associated_thoughts = []
            for assoc_id in direct_associations:
                assoc_thought = next((t for t in self.thought_history if t.id == assoc_id), None)
                if assoc_thought:
                    associated_thoughts.append(assoc_thought)
            
            if associated_thoughts:
                connections.append(f"Directly associated with {len(associated_thoughts)} previous thoughts")
        
        # 2. Find semantic connections (word overlap)
        semantic_connections = []
        
        for prev_thought in self.thought_history[-10:]:  # Recent thoughts
            if prev_thought.id == thought_id:
                continue  # Skip self
            
            prev_content = prev_thought.content.lower()
            prev_words = set(w for w in prev_content.split() if len(w) > 3)
            
            # Calculate word overlap
            common_words = content_words.intersection(prev_words)
            overlap_score = len(common_words) / max(1, min(len(content_words), len(prev_words)))
            
            if overlap_score > 0.3:  # Significant overlap
                summary = prev_thought.content[:30] + "..." if len(prev_thought.content) > 30 else prev_thought.content
                semantic_connections.append((prev_thought.id, overlap_score, summary))
        
        # Sort and add top semantic connections
        if semantic_connections:
            semantic_connections.sort(key=lambda x: x[1], reverse=True)
            top_connections = semantic_connections[:2]  # Limit to 2
            
            for _, score, summary in top_connections:
                connections.append(f"Semantic connection ({score:.2f}): \"{summary}\"")
        
        # 3. Find thought type connections
        type_connections = []
        
        for prev_thought in self.thought_history[-10:]:
            if prev_thought.id == thought_id:
                continue
            
            if prev_thought.type == thought.type:
                type_connections.append(prev_thought)
        
        if type_connections:
            connections.append(f"Shares {thought.type.value} thought type with {len(type_connections)} recent thoughts")
        
        # 4. Process-based connections
        process_connections = []
        
        for prev_thought in self.thought_history[-10:]:
            if prev_thought.id == thought_id:
                continue
                
            # Check for process overlap
            common_processes = set(thought.processes).intersection(set(prev_thought.processes))
            if common_processes and len(common_processes) >= 2:
                process_str = ", ".join(p.value for p in common_processes)
                process_connections.append((prev_thought.id, process_str))
        
        if process_connections:
            connections.append(f"Shares cognitive processes with {len(process_connections)} recent thoughts")
        
        # 5. Contextual connections
        if thought.context and thought.context.get("context_id"):
            context_id = thought.context["context_id"]
            context_thoughts = [t for t in self.thought_history if 
                            t.id != thought_id and 
                            t.context.get("context_id") == context_id]
            
            if context_thoughts:
                connections.append(f"Shares context with {len(context_thoughts)} other thoughts")
        
        # Limit connections based on capability
        max_connections = int(1 + 4 * connection_ability)
        return connections[:max_connections]
    
    def _generate_meta_thoughts(self, patterns: List[str], insights: List[str]) -> List[str]:
        """
        Generate meta-thoughts about patterns and insights.
        
        This creates higher-order reflections on identified patterns and insights,
        capable of varying sophistication based on metacognitive development.
        
        Args:
            patterns: Identified patterns
            insights: Derived insights/implications
            
        Returns:
            List of meta-thoughts
        """
        meta_thoughts = []
        
        # Meta-thought generation depends heavily on metacognition
        metacognition_level = self.cognitive_capabilities["metacognition"]
        if metacognition_level < 0.4:
            return []  # Limited metacognitive ability
        
        # 1. Basic pattern reflection
        if patterns:
            # Simple pattern observations for lower metacognition
            if metacognition_level < 0.6:
                meta_thoughts.append(f"I notice I'm thinking about similar things repeatedly")
            else:
                # More sophisticated pattern awareness
                if len(patterns) > 1:
                    meta_thoughts.append(f"I observe multiple patterns in my recent thinking: {'; '.join(patterns[:2])}")
                else:
                    meta_thoughts.append(f"I recognize a pattern in my thinking: {patterns[0]}")
        
        # 2. Basic insight reflection
        if insights:
            if metacognition_level < 0.6:
                meta_thoughts.append(f"These thoughts might lead to important conclusions")
            else:
                meta_thoughts.append(f"The implications of my thoughts suggest {insights[0].lower()}")
        
        # 3. Thinking process reflection
        process_reflection_chance = metacognition_level * 0.8
        if random.random() < process_reflection_chance:
            # Choose reflection based on metacognition level
            if metacognition_level < 0.6:
                reflections = [
                    "I'm noticing how my thoughts connect to each other",
                    "I can see how one thought leads to another",
                    "My thinking seems to follow certain themes"
                ]
            else:
                reflections = [
                    "I'm observing the structure and evolution of my thought processes",
                    "I notice how my cognitive processes influence the content of my thoughts",
                    "The patterns in my thinking reveal my current cognitive priorities and biases",
                    "My thought sequences show characteristic patterns of association and development"
                ]
            meta_thoughts.append(random.choice(reflections))
        
        # 4. Bias awareness (higher metacognition only)
        if metacognition_level > 0.7:
            bias_awareness_chance = (metacognition_level - 0.7) * 2  # Scales from 0-0.6
            if random.random() < bias_awareness_chance:
                bias_reflections = [
                    "I might be influenced by cognitive biases in my thinking pattern",
                    "My recent thoughts may reflect confirmation bias for existing beliefs",
                    "I should consider whether availability bias is affecting my judgments",
                    "My thinking might be anchored to initial concepts, limiting exploration"
                ]
                meta_thoughts.append(random.choice(bias_reflections))
        
        # 5. Advanced metacognitive reflection (highest levels only)
        if metacognition_level > 0.8:
            advanced_reflection_chance = (metacognition_level - 0.8) * 3  # Scales from 0-0.6
            if random.random() < advanced_reflection_chance:
                advanced_reflections = [
                    "The recursive nature of these meta-thoughts demonstrates progressive metacognitive development",
                    "My ability to reflect on my own thought patterns represents a high-level cognitive function",
                    "This multi-level awareness of my own thinking processes suggests developing metacognitive maturity",
                    "The integration of self-reflection with pattern recognition indicates cognitive integration"
                ]
                meta_thoughts.append(random.choice(advanced_reflections))
        
        # Limit based on metacognitive capacity
        meta_thought_capacity = int(1 + 3 * metacognition_level)
        return meta_thoughts[:meta_thought_capacity]
    
    def _handle_distraction(self, distraction: str) -> None:
        """
        Handle a distraction and its effect on attention.
        
        This models how distractions affect the attentional focus, with
        developmental differences in distraction handling capability.
        
        Args:
            distraction: The distraction content
        """
        # Measure distraction severity based on length and content
        severity = min(1.0, len(distraction) / 50)  # Longer distractions are more severe
        
        # Check for emotionally charged content that might be more distracting
        emotional_words = ["urgent", "important", "danger", "exciting", "worry", 
                        "problem", "emergency", "critical", "failure", "success"]
        
        for word in emotional_words:
            if word in distraction.lower():
                severity += 0.1  # Emotional content increases severity
                
        severity = min(1.0, severity)  # Cap at 1.0
        
        # Calculate distraction resistance based on cognitive development
        # Abstract thinking and metacognition help resist distractions
        distraction_resistance = (
            0.7 * self.cognitive_capabilities["metacognition"] +
            0.3 * self.cognitive_capabilities["abstraction"]
        )
        
        # Calculate focus impact, mitigated by resistance
        focus_impact = severity * (1.0 - distraction_resistance)
        
        # Apply focus impact with some randomness
        random_factor = 0.8 + (random.random() * 0.4)  # 0.8 to 1.2
        final_impact = focus_impact * random_factor
        
        # Update focus level
        self.attention_state.focus_level = max(0.1, self.attention_state.focus_level - final_impact)
        
        # Add distraction to current distractions
        if len(self.attention_state.distractions) >= 5:
            self.attention_state.distractions.pop(0)  # Remove oldest distraction
        
        # Determine how much of the distraction is processed based on severity
        processed_distraction = distraction
        if len(distraction) > 20 and focus_impact > 0.3:
            # Only partially process long distractions when focus impact is high
            truncated_length = max(10, int(len(distraction) * (1.0 - focus_impact)))
            processed_distraction = distraction[:truncated_length] + "..."
        
        self.attention_state.distractions.append(processed_distraction)
        
        # Reset sustained focus duration if focus drops significantly
        if final_impact > 0.3:
            self.attention_state.sustained_focus_duration = 0
            
        logger.debug(f"Handled distraction with severity {severity:.2f}, focus impact: {final_impact:.2f}, new focus: {self.attention_state.focus_level:.2f}")