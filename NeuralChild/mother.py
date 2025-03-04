# mother.py
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import logging
from pathlib import Path
import os

# Import from your existing modules
from llm_module import LLMClient, Message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MotherModule")

# ========== PYDANTIC MODELS ==========

class VerbalResponse(BaseModel):
    """Model for the mother's verbal response"""
    text: str = Field(..., description="The actual words spoken by the mother")
    tone: str = Field(..., description="How the mother says the words (gentle, firm, excited, etc.)")
    complexity_level: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Language complexity relative to child's development"
    )

class EmotionalState(BaseModel):
    """Model for the mother's emotional state"""
    primary_emotion: str = Field(..., description="Main emotion felt by the mother")
    intensity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Intensity of the primary emotion"
    )
    secondary_emotion: Optional[str] = Field(
        None, 
        description="Optional secondary emotion if feelings are mixed"
    )
    patience_level: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Current level of patience"
    )

class NonVerbalResponse(BaseModel):
    """Model for the mother's non-verbal behaviors"""
    physical_actions: List[str] = Field(
        ..., 
        description="Physical actions taken by the mother"
    )
    facial_expression: str = Field(
        ..., 
        description="Mother's facial expression"
    )
    proximity: str = Field(
        ..., 
        description="Physical proximity to the child"
    )

class VocabularyItem(BaseModel):
    """Model for vocabulary items being introduced"""
    word: str = Field(..., description="The new word being taught")
    simple_definition: str = Field(..., description="Child-appropriate definition")
    example_usage: Optional[str] = Field(None, description="Example using the word")

class ConceptItem(BaseModel):
    """Model for new concepts being taught"""
    concept_name: str = Field(..., description="Name of the concept")
    explanation: str = Field(..., description="Child-appropriate explanation")
    relevance: Optional[str] = Field(None, description="Why this concept matters now")

class CorrectionItem(BaseModel):
    """Model for gentle corrections to child's misunderstandings"""
    misunderstanding: str = Field(..., description="What the child misunderstood")
    correction: str = Field(..., description="The gentle correction")
    approach: str = Field(..., description="How this is being corrected (direct, indirect)")

class TeachingElements(BaseModel):
    """Model for the teaching aspects of the response"""
    vocabulary: List[VocabularyItem] = Field(
        default_factory=list, 
        description="New vocabulary being introduced"
    )
    concepts: List[ConceptItem] = Field(
        default_factory=list, 
        description="New concepts being taught"
    )
    values: List[str] = Field(
        default_factory=list, 
        description="Moral/social values being conveyed"
    )
    corrections: List[CorrectionItem] = Field(
        default_factory=list, 
        description="Gentle corrections to misunderstandings"
    )

class ChildPerception(BaseModel):
    """Model for how the mother perceives the child"""
    child_emotion: str = Field(..., description="Mother's perception of child's emotion")
    child_needs: List[str] = Field(..., description="What mother thinks child needs")
    misinterpretations: Optional[List[str]] = Field(
        None, 
        description="Potential misinterpretations (for realism)"
    )

class ParentingApproach(BaseModel):
    """Model for the parenting approach taken"""
    intention: str = Field(..., description="What mother is trying to accomplish")
    approach: str = Field(..., description="Strategy being used")
    consistency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Consistency with previous responses"
    )
    adaptation_to_development: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How well matched to child's development"
    )

class ContextAwareness(BaseModel):
    """Model for mother's awareness of context"""
    references_previous_interactions: Optional[List[str]] = Field(
        None, 
        description="References to previous interactions"
    )
    environment_factors: Optional[List[str]] = Field(
        None, 
        description="Relevant environmental factors"
    )
    recognizes_progress: Optional[bool] = Field(
        None, 
        description="Whether mother notices child's development"
    )

class MotherResponse(BaseModel):
    """Complete model for a mother's response"""
    verbal: VerbalResponse
    emotional: EmotionalState
    non_verbal: NonVerbalResponse
    teaching: TeachingElements
    perception: ChildPerception
    parenting: ParentingApproach
    context_awareness: ContextAwareness

class ChildState(BaseModel):
    """Current state of the neural child that mother can observe"""
    message: str = Field(..., description="What the child actually said/communicated")
    apparent_emotion: str = Field(..., description="Observable emotional state")
    vocabulary_size: int = Field(..., ge=0, description="Current vocabulary size")
    age_days: float = Field(..., ge=0.0, description="Age in simulated days")
    recent_concepts_learned: List[str] = Field(
        default_factory=list, 
        description="Recently acquired concepts"
    )
    attention_span: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Current attention capacity"
    )

class InteractionHistory(BaseModel):
    """Simplified history of recent mother-child interactions"""
    recent_exchanges: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Recent back-and-forth exchanges"
    )
    current_activity: Optional[str] = Field(
        None, 
        description="Context of the current interaction"
    )
    time_since_last_interaction: Optional[float] = Field(
        None, 
        description="Minutes since the last interaction"
    )

class MotherConfig(BaseModel):
    """Configuration for the mother's personality and parenting style"""
    personality_traits: Dict[str, float] = Field(
        default_factory=dict, 
        description="Key personality traits with intensity levels"
    )
    teaching_priority: List[str] = Field(
        default_factory=list, 
        description="What this mother prioritizes teaching"
    )
    communication_style: str = Field(
        "balanced", 
        description="Overall communication approach"
    )
    emotional_expressiveness: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="How emotionally expressive mother is"
    )
    patience_baseline: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Baseline patience level"
    )
    
    @field_validator('personality_traits')
    @classmethod
    def validate_traits(cls, traits):
        """Validate that personality traits have valid intensity levels"""
        for trait, level in traits.items():
            if not (0.0 <= level <= 1.0):
                raise ValueError(f"Trait intensity for {trait} must be between 0 and 1")
        return traits

# ========== MOTHER IMPLEMENTATION ==========

class Mother:
    """The mother component of the neural child system"""
    
    def __init__(
        self, 
        llm_client: LLMClient,
        config_path: Optional[Path] = None,
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        history_size: int = 10
    ):
        """Initialize the mother component with LLM client and configuration"""
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_size = history_size
        self.interaction_history = InteractionHistory(recent_exchanges=[])
        
        # Load or create default configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.config = MotherConfig(**config_data)
        else:
            # Default mother configuration
            self.config = MotherConfig(
                personality_traits={
                    "warmth": 0.85,
                    "creativity": 0.7,
                    "strictness": 0.4,
                    "playfulness": 0.75,
                    "protectiveness": 0.8
                },
                teaching_priority=[
                    "emotional_regulation",
                    "language_development",
                    "social_skills",
                    "curiosity",
                    "independence"
                ],
                communication_style="nurturing",
                emotional_expressiveness=0.8,
                patience_baseline=0.85
            )
        
        # Create schema for structured completion
        self._create_response_schema()
        
        logger.info(f"Mother component initialized with {model} model")
    
    def _create_response_schema(self) -> None:
        """Create the JSON schema for structured LLM output"""
        # Create a schema for the model output
        # This uses the structure from MotherResponse but simplified for the LLM
        self.response_schema = {
            "name": "mother_response",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "verbal": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tone": {"type": "string"},
                            "complexity_level": {"type": "number"}
                        },
                        "required": ["text", "tone", "complexity_level"]
                    },
                    "emotional": {
                        "type": "object",
                        "properties": {
                            "primary_emotion": {"type": "string"},
                            "intensity": {"type": "number"},
                            "secondary_emotion": {"type": ["string", "null"]},
                            "patience_level": {"type": "number"}
                        },
                        "required": ["primary_emotion", "intensity", "patience_level"]
                    },
                    "non_verbal": {
                        "type": "object",
                        "properties": {
                            "physical_actions": {"type": "array", "items": {"type": "string"}},
                            "facial_expression": {"type": "string"},
                            "proximity": {"type": "string"}
                        },
                        "required": ["physical_actions", "facial_expression", "proximity"]
                    },
                    "teaching": {
                        "type": "object",
                        "properties": {
                            "vocabulary": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "word": {"type": "string"},
                                        "simple_definition": {"type": "string"},
                                        "example_usage": {"type": ["string", "null"]}
                                    },
                                    "required": ["word", "simple_definition"]
                                }
                            },
                            "concepts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "concept_name": {"type": "string"},
                                        "explanation": {"type": "string"},
                                        "relevance": {"type": ["string", "null"]}
                                    },
                                    "required": ["concept_name", "explanation"]
                                }
                            },
                            "values": {"type": "array", "items": {"type": "string"}},
                            "corrections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "misunderstanding": {"type": "string"},
                                        "correction": {"type": "string"},
                                        "approach": {"type": "string"}
                                    },
                                    "required": ["misunderstanding", "correction", "approach"]
                                }
                            }
                        },
                        "required": ["vocabulary", "concepts", "values", "corrections"]
                    },
                    "perception": {
                        "type": "object",
                        "properties": {
                            "child_emotion": {"type": "string"},
                            "child_needs": {"type": "array", "items": {"type": "string"}},
                            "misinterpretations": {"type": ["array", "null"], "items": {"type": "string"}}
                        },
                        "required": ["child_emotion", "child_needs"]
                    },
                    "parenting": {
                        "type": "object",
                        "properties": {
                            "intention": {"type": "string"},
                            "approach": {"type": "string"},
                            "consistency": {"type": "number"},
                            "adaptation_to_development": {"type": "number"}
                        },
                        "required": ["intention", "approach", "consistency", "adaptation_to_development"]
                    },
                    "context_awareness": {
                        "type": "object",
                        "properties": {
                            "references_previous_interactions": {"type": ["array", "null"], "items": {"type": "string"}},
                            "environment_factors": {"type": ["array", "null"], "items": {"type": "string"}},
                            "recognizes_progress": {"type": ["boolean", "null"]}
                        }
                    }
                },
                "required": ["verbal", "emotional", "non_verbal", "teaching", "perception", "parenting", "context_awareness"]
            }
        }
    
    def _create_mother_prompt(self, child_state: ChildState) -> List[Message]:
        """Create the prompt for the mother LLM based on child state and history"""
        # System prompt establishes the mother role and guidelines
        system_prompt = f"""You are a mother interacting with your young neural child who is still developing. 
You should respond like a real mother would - not as an AI or assistant.

Your child's current state:
- Age: {child_state.age_days:.1f} days (simulated development time)
- Vocabulary size: {child_state.vocabulary_size} words
- Current emotional state appears to be: {child_state.apparent_emotion}
- Attention span level: {child_state.attention_span:.2f} (0-1 scale)

Your personality and parenting style:
{json.dumps(self.config.model_dump(), indent=2)}

Recent interaction history:
{json.dumps(self.interaction_history.model_dump(), indent=2)}

Important guidelines:
1. Respond naturally as a mother would, with appropriate emotional range
2. Match your language complexity to your child's development level ({child_state.vocabulary_size} words)
3. Your child is still learning language, so expect simple or incomplete sentences
4. Be supportive but establish age-appropriate boundaries
5. Occasionally show human imperfection like mild impatience (be realistic!)
6. Introduce new vocabulary and concepts gradually
7. You can only respond to what you can observe - you're not a mind reader

You must structure your response exactly according to the provided schema.
Remember that real mothers aren't perfect - include realistic variations in patience and occasional misinterpretations.
"""

        # User message contains what the child actually said
        user_message = f"Your child says to you: \"{child_state.message}\""
        
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]
    
    def respond_to_child(self, child_state: ChildState) -> MotherResponse:
        """Generate a mother's response to the child's current state"""
        logger.info(f"Generating mother response to: {child_state.message}")
        
        # Prepare the prompt
        messages = self._create_mother_prompt(child_state)
        
        try:
            # Get structured response from LLM
            response_data = self.llm_client.structured_completion(
                messages=messages,
                json_schema=self.response_schema,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response into our Pydantic model
            mother_response = MotherResponse(**response_data)
            
            # Update interaction history
            self._update_history(child_state, mother_response)
            
            logger.info(f"Generated response: {mother_response.verbal.text}")
            return mother_response
            
        except Exception as e:
            logger.error(f"Error generating mother response: {str(e)}")
            # Create a fallback response
            return self._create_fallback_response(child_state)
    
    def _update_history(self, child_state: ChildState, response: MotherResponse) -> None:
        """Update the interaction history with the latest exchange"""
        # Create a record of this exchange
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "child_message": child_state.message,
            "mother_response": response.verbal.text,
            "apparent_emotion": child_state.apparent_emotion,
            "mother_emotion": response.emotional.primary_emotion
        }
        
        # Add to history and maintain max size
        self.interaction_history.recent_exchanges.append(exchange)
        if len(self.interaction_history.recent_exchanges) > self.history_size:
            self.interaction_history.recent_exchanges.pop(0)
        
        # Update other history fields
        self.interaction_history.time_since_last_interaction = 0
    
    def _create_fallback_response(self, child_state: ChildState) -> MotherResponse:
        """Create a simple fallback response if LLM fails"""
        logger.warning("Using fallback response mechanism")
        
        # Simple fallback logic based on child's apparent emotion
        if child_state.apparent_emotion.lower() in ["sad", "upset", "crying"]:
            verbal_text = "There, there. It's going to be okay. Mommy's here."
            emotion = "concern"
            action = "gives a gentle hug"
        elif child_state.apparent_emotion.lower() in ["happy", "excited", "joyful"]:
            verbal_text = "You look so happy! That makes me happy too!"
            emotion = "joy"
            action = "smiles warmly"
        else:
            verbal_text = "I'm here with you. What would you like to do now?"
            emotion = "attentive"
            action = "kneels down to child's level"
        
        # Construct a basic response with valid values
        return MotherResponse(
            verbal=VerbalResponse(
                text=verbal_text,
                tone="gentle",
                complexity_level=0.3  # Fixed: Was 2 in original code, maximum allowed is 1.0
            ),
            emotional=EmotionalState(
                primary_emotion=emotion,
                intensity=0.7,
                secondary_emotion=None,
                patience_level=0.9
            ),
            non_verbal=NonVerbalResponse(
                physical_actions=[action],
                facial_expression="warm, attentive",
                proximity="close"
            ),
            teaching=TeachingElements(
                vocabulary=[],
                concepts=[],
                values=["security"],
                corrections=[]
            ),
            perception=ChildPerception(
                child_emotion=child_state.apparent_emotion,
                child_needs=["attention", "reassurance"],
                misinterpretations=None
            ),
            parenting=ParentingApproach(
                intention="provide security and reassurance",
                approach="nurturing",
                consistency=0.9,
                adaptation_to_development=0.8
            ),
            context_awareness=ContextAwareness(
                references_previous_interactions=None,
                environment_factors=None,
                recognizes_progress=None
            )
        )
    
    def adjust_for_developmental_stage(self, response: MotherResponse, vocabulary_size: int) -> MotherResponse:
        """Adjust a mother's response to better match child's developmental stage"""
        # This would modify complexity of language and teaching elements based on vocabulary
        adjusted = response.model_copy(deep=True)
        
        # Simplified developmental adjustments (could be much more sophisticated)
        if vocabulary_size < 50:  # Very early stage
            adjusted.verbal.complexity_level = min(adjusted.verbal.complexity_level, 0.2)
            adjusted.verbal.text = self._simplify_text(adjusted.verbal.text)
            adjusted.teaching.vocabulary = adjusted.teaching.vocabulary[:1]  # Limit new words
        elif vocabulary_size < 200:  # Early stage
            adjusted.verbal.complexity_level = min(adjusted.verbal.complexity_level, 0.4)
            adjusted.teaching.vocabulary = adjusted.teaching.vocabulary[:2]  # Limit new words
        elif vocabulary_size < 500:  # Intermediate stage
            adjusted.verbal.complexity_level = min(adjusted.verbal.complexity_level, 0.6)
        
        return adjusted
    
    def _simplify_text(self, text: str) -> str:
        """Simplify language for very early developmental stages"""
        # Very simple implementation - in a real system this would be more sophisticated
        # Could use NLP techniques to simplify grammar, vocabulary, etc.
        words = text.split()
        if len(words) > 5:
            # Break into shorter phrases
            simplified = []
            for i in range(0, len(words), 4):
                simplified.append(" ".join(words[i:i+4]))
            return ". ".join(simplified)
        return text
    
    def save_config(self, path: Path) -> None:
        """Save the current mother configuration"""
        with open(path, 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2)
            logger.info(f"Mother configuration saved to {path}")

# ========== USAGE EXAMPLE ==========

def create_example_mother(api_url: str = "http://192.168.2.12:1234"):
    """Create an example mother instance with realistic configuration"""
    llm_client = LLMClient(base_url=api_url)
    
    mother = Mother(
        llm_client=llm_client,
        model="qwen2.5-7b-instruct",
        temperature=0.75,  # Slightly higher for more natural variation
        max_tokens=1000,
        history_size=10
    )
    
    # Customize mother's personality if desired
    mother.config.personality_traits["warmth"] = 0.9
    mother.config.personality_traits["playfulness"] = 0.8
    mother.config.teaching_priority = [
        "emotional_intelligence",
        "language_development",
        "curiosity",
        "confidence",
        "social_connection"
    ]
    
    return mother

# Example usage
if __name__ == "__main__":
    # Quick test to demonstrate usage
    mother = create_example_mother()
    
    example_state = ChildState(
        message="Mama! Look... ball!",
        apparent_emotion="excited",
        vocabulary_size=35,
        age_days=5.2,
        recent_concepts_learned=["ball", "red", "big"],
        attention_span=0.4
    )
    
    response = mother.respond_to_child(example_state)
    print(f"Mother says: {response.verbal.text}")
    print(f"Emotion: {response.emotional.primary_emotion} (intensity: {response.emotional.intensity})")
    print(f"Actions: {', '.join(response.non_verbal.physical_actions)}")
    
    if response.teaching.vocabulary:
        print("\nTeaching new words:")
        for item in response.teaching.vocabulary:
            print(f"- {item.word}: {item.simple_definition}")