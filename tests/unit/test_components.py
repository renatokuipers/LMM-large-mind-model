"""
Unit tests for the NeuralChild components.

This module contains tests for the various neural components that make up
the child's mind.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json

from neuralchild.components.memory import MemorySystem
from neuralchild.components.language import LanguageComponent
from neuralchild.components.emotional import EmotionalComponent
from neuralchild.components.consciousness import ConsciousnessComponent
from neuralchild.components.social import SocialComponent
from neuralchild.components.cognitive import CognitiveComponent

from neuralchild.utils.data_types import (
    DevelopmentalStage, Emotion, EmotionType,
    Memory, EpisodicMemory, SemanticMemory, MemoryType
)


class TestMemorySystem(unittest.TestCase):
    """Test case for the Memory System component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.faiss_index_path = os.path.join(self.temp_dir, "faiss_indexes")
        self.vector_db_path = os.path.join(self.temp_dir, "vector_db")
        
        # Create memory system with test paths
        self.memory_system = MemorySystem(
            faiss_index_path=self.faiss_index_path,
            vector_db_path=self.vector_db_path,
            use_gpu=False
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directories
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test initialization of Memory System."""
        self.assertEqual(self.memory_system.faiss_index_path, self.faiss_index_path)
        self.assertEqual(self.memory_system.vector_db_path, self.vector_db_path)
        self.assertFalse(self.memory_system.use_gpu)
        self.assertIsNotNone(self.memory_system.episodic_index)
        self.assertIsNotNone(self.memory_system.semantic_index)
    
    def test_store_and_retrieve_episodic_memory(self):
        """Test storing and retrieving episodic memories."""
        # Create a test memory
        memory = EpisodicMemory(
            id="test_memory",
            type=MemoryType.EPISODIC,
            event_description="Playing with blocks",
            emotional_valence=0.8,
            associated_emotions=[Emotion(type=EmotionType.JOY, intensity=0.7)]
        )
        
        # Mock embedding function (normally would call LLM)
        with patch.object(self.memory_system, '_get_embedding', return_value=[0.1] * 384):
            # Store memory
            self.memory_system.store_memory(memory)
            
            # Retrieve similar memories
            memories = self.memory_system.retrieve_similar_memories(
                query="Playing with toys",
                limit=5
            )
            
            # Should find the stored memory
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].id, "test_memory")
    
    def test_store_and_retrieve_semantic_memory(self):
        """Test storing and retrieving semantic memories."""
        # Create a test memory
        memory = SemanticMemory(
            id="test_concept",
            type=MemoryType.SEMANTIC,
            concept="Dog",
            definition="A four-legged animal that barks",
            related_concepts=["Pet", "Animal"]
        )
        
        # Mock embedding function (normally would call LLM)
        with patch.object(self.memory_system, '_get_embedding', return_value=[0.1] * 384):
            # Store memory
            self.memory_system.store_memory(memory)
            
            # Retrieve similar memories
            memories = self.memory_system.retrieve_similar_memories(
                query="Animals that make noise",
                memory_type=MemoryType.SEMANTIC,
                limit=5
            )
            
            # Should find the stored memory
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].id, "test_concept")
    
    def test_memory_decay(self):
        """Test that memories decay over time."""
        # Create a test memory
        memory = EpisodicMemory(
            id="decay_test",
            type=MemoryType.EPISODIC,
            event_description="A fading memory",
            emotional_valence=0.5,
            strength=1.0,  # Start at full strength
            decay_rate=0.5  # High decay rate for testing
        )
        
        # Mock embedding function
        with patch.object(self.memory_system, '_get_embedding', return_value=[0.1] * 384):
            # Store memory
            self.memory_system.store_memory(memory)
            
            # Apply decay
            self.memory_system.apply_memory_decay()
            
            # Retrieve and check strength
            memories = self.memory_system.retrieve_memories_by_id(["decay_test"])
            self.assertEqual(len(memories), 1)
            # Strength should be reduced by decay_rate
            self.assertLess(memories[0].strength, 1.0)


class TestLanguageComponent(unittest.TestCase):
    """Test case for the Language Component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.language_component = LanguageComponent()
    
    def test_init(self):
        """Test initialization of Language Component."""
        self.assertEqual(len(self.language_component.vocabulary), 0)
        self.assertEqual(self.language_component.grammar_complexity, 0.0)
        self.assertEqual(self.language_component.phonological_development, 0.0)
        self.assertEqual(self.language_component.semantic_understanding, 0.0)
        self.assertEqual(self.language_component.pragmatic_understanding, 0.0)
    
    def test_process_language_input_infancy(self):
        """Test processing language input during infancy."""
        result = self.language_component.process_language_input(
            "Hello little one!",
            developmental_stage=DevelopmentalStage.INFANCY,
            age_months=3
        )
        
        # During early infancy, should not understand words
        self.assertDictEqual(result, {
            "understood_words": [],
            "detected_sentiment": "positive",
            "response_type": "babble"
        })
    
    def test_process_language_input_early_childhood(self):
        """Test processing language input during early childhood."""
        # Add some words to vocabulary first
        self.language_component.vocabulary = {
            "hello": {"understanding_level": 0.8},
            "toy": {"understanding_level": 0.7}
        }
        
        result = self.language_component.process_language_input(
            "Hello, would you like to play with this toy?",
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            age_months=30
        )
        
        # Should recognize known words
        self.assertIn("hello", result["understood_words"])
        self.assertIn("toy", result["understood_words"])
        self.assertEqual(result["detected_sentiment"], "positive")
        self.assertEqual(result["response_type"], "simple_phrase")
    
    def test_vocabulary_growth(self):
        """Test vocabulary growth over time."""
        # Initial vocabulary is empty
        self.assertEqual(len(self.language_component.vocabulary), 0)
        
        # Process several inputs
        for _ in range(5):
            self.language_component.process_language_input(
                "Hello little one. How are you today? Let's play with blocks.",
                developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
                age_months=30
            )
        
        # Should have learned some words
        self.assertGreater(len(self.language_component.vocabulary), 0)
        
        # Verify vocabulary size getter
        vocab_size = self.language_component.get_vocabulary_size()
        self.assertEqual(vocab_size, len(self.language_component.vocabulary))


class TestEmotionalComponent(unittest.TestCase):
    """Test case for the Emotional Component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.emotional_component = EmotionalComponent()
    
    def test_init(self):
        """Test initialization of Emotional Component."""
        # Check initial emotions
        self.assertGreater(len(self.emotional_component.current_emotions), 0)
        
        # Check temperament values
        self.assertGreater(self.emotional_component.temperament_reactivity, 0)
        self.assertLess(self.emotional_component.temperament_reactivity, 1)
        
        self.assertGreater(self.emotional_component.temperament_regulation, 0)
        self.assertLess(self.emotional_component.temperament_regulation, 1)
        
        self.assertGreater(self.emotional_component.temperament_sociability, 0)
        self.assertLess(self.emotional_component.temperament_sociability, 1)
    
    def test_process_emotional_input(self):
        """Test processing emotional input."""
        # Create test emotions
        input_emotions = [
            Emotion(type=EmotionType.JOY, intensity=0.8),
            Emotion(type=EmotionType.SURPRISE, intensity=0.4)
        ]
        
        # Process emotions
        result = self.emotional_component.process_emotional_input(
            input_emotions,
            developmental_stage=DevelopmentalStage.INFANCY
        )
        
        # Check response emotions
        self.assertGreater(len(result), 0)
        
        # Emotion types should be influenced by input
        emotion_types = [e.type for e in result]
        self.assertIn(EmotionType.JOY, emotion_types)
    
    def test_update_emotional_development(self):
        """Test updating emotional development based on stage."""
        # Initial regulation
        initial_regulation = self.emotional_component.emotional_regulation
        
        # Update for adolescence (should have higher regulation)
        self.emotional_component.update_emotional_development(
            developmental_stage=DevelopmentalStage.ADOLESCENCE
        )
        
        # Regulation should increase
        self.assertGreater(self.emotional_component.emotional_regulation, initial_regulation)
        
        # Verify getter
        regulation = self.emotional_component.get_emotional_regulation()
        self.assertEqual(regulation, self.emotional_component.emotional_regulation)


class TestConsciousnessComponent(unittest.TestCase):
    """Test case for the Consciousness Component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.consciousness_component = ConsciousnessComponent()
    
    def test_init(self):
        """Test initialization of Consciousness Component."""
        self.assertEqual(self.consciousness_component.self_awareness, 0.1)
        self.assertEqual(self.consciousness_component.theory_of_mind, 0.0)
        self.assertEqual(self.consciousness_component.reflective_thinking, 0.0)
        self.assertEqual(self.consciousness_component.autobiographical_awareness, 0.0)
        self.assertEqual(self.consciousness_component.metacognition, 0.0)
    
    def test_update_consciousness_development(self):
        """Test updating consciousness development based on stage."""
        # Initial state
        initial_self_awareness = self.consciousness_component.self_awareness
        
        # Update for early childhood
        changes = self.consciousness_component.update_consciousness_development(
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD,
            age_months=36
        )
        
        # Self-awareness should increase
        self.assertGreater(self.consciousness_component.self_awareness, initial_self_awareness)
        self.assertIn("self_awareness", changes)


class TestSocialComponent(unittest.TestCase):
    """Test case for the Social Component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.social_component = SocialComponent()
    
    def test_init(self):
        """Test initialization of Social Component."""
        self.assertEqual(self.social_component.social_awareness, 0.1)
        self.assertEqual(self.social_component.relationship_understanding, 0.0)
        self.assertEqual(self.social_component.social_norm_comprehension, 0.0)
        self.assertEqual(len(self.social_component.interaction_history), 0)
    
    def test_update_social_development(self):
        """Test updating social development based on stage."""
        # Initial social awareness
        initial_awareness = self.social_component.social_awareness
        
        # Update for middle childhood
        changes = self.social_component.update_social_development(
            developmental_stage=DevelopmentalStage.MIDDLE_CHILDHOOD
        )
        
        # Social awareness should increase
        self.assertGreater(self.social_component.social_awareness, initial_awareness)
        self.assertIn("social_awareness", changes)
        
        # Verify getter
        awareness = self.social_component.get_social_awareness()
        self.assertEqual(awareness, self.social_component.social_awareness)
    
    def test_process_social_interaction(self):
        """Test processing a social interaction."""
        # Process a social interaction
        result = self.social_component.process_social_interaction(
            "Let's play together with blocks!",
            developmental_stage=DevelopmentalStage.EARLY_CHILDHOOD
        )
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn("social_elements", result)
        
        # History should be updated
        self.assertEqual(len(self.social_component.interaction_history), 1)


class TestCognitiveComponent(unittest.TestCase):
    """Test case for the Cognitive Component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cognitive_component = CognitiveComponent()
    
    def test_init(self):
        """Test initialization of Cognitive Component."""
        self.assertEqual(self.cognitive_component.object_permanence, 0.1)
        self.assertEqual(self.cognitive_component.conservation, 0.0)
        self.assertEqual(self.cognitive_component.abstract_thinking, 0.0)
        self.assertEqual(self.cognitive_component.problem_solving, 0.0)
    
    def test_update_cognitive_development(self):
        """Test updating cognitive development based on stage."""
        # Initial cognitive abilities
        initial_abstract = self.cognitive_component.abstract_thinking
        
        # Update for adolescence
        changes = self.cognitive_component.update_cognitive_development(
            developmental_stage=DevelopmentalStage.ADOLESCENCE
        )
        
        # Abstract thinking should increase
        self.assertGreater(self.cognitive_component.abstract_thinking, initial_abstract)
        self.assertIn("abstract_thinking", changes)


if __name__ == '__main__':
    unittest.main() 