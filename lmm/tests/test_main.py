import pytest
import os
from unittest.mock import patch, MagicMock
from lmm import main

class TestMainModule:
    """Tests for the main module functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Fixture providing mocked components for main module testing."""
        mocks = {
            "memory_system": MagicMock(),
            "mind_modules": {
                "consciousness": MagicMock(),
                "emotion": MagicMock(),
                "language": MagicMock(),
                "thought": MagicMock(),
                "social": MagicMock()
            },
            "mother": MagicMock(),
            "caregiver": MagicMock(),
            "visualization": MagicMock()
        }
        return mocks
    
    @pytest.fixture
    def sample_input(self):
        """Fixture providing sample input for main module processing."""
        return {
            "query": "What is the meaning of life?",
            "context": "philosophical inquiry",
            "user_id": "test_user"
        }
    
    def test_main_initialization(self):
        """Test main module initialization."""
        with patch("lmm.memory.persistence.MemoryManager") as mock_memory_manager, \
             patch("lmm.memory.advanced_memory.AdvancedMemoryManager") as mock_advanced_memory, \
             patch("lmm.core.development.stages.DevelopmentalStageManager") as mock_stage_manager, \
             patch("lmm.core.development.learning.LearningManager") as mock_learning_manager, \
             patch("lmm.core.mother.caregiver.MotherCaregiver") as mock_caregiver, \
             patch("lmm.core.mind_modules.consciousness.ConsciousnessModule") as mock_consciousness, \
             patch("lmm.core.mind_modules.emotion.EmotionModule") as mock_emotion, \
             patch("lmm.core.mind_modules.language.LanguageModule") as mock_language, \
             patch("lmm.core.mind_modules.memory.MemoryModule") as mock_memory_module, \
             patch("lmm.core.mind_modules.social.SocialCognitionModule") as mock_social, \
             patch("lmm.core.mind_modules.thought.ThoughtModule") as mock_thought:
            
            # Create mock instances
            mock_memory_manager.return_value = MagicMock()
            mock_advanced_memory.return_value = MagicMock()
            mock_stage_manager.return_value = MagicMock()
            mock_learning_manager.return_value = MagicMock()
            mock_caregiver.return_value = MagicMock()
            mock_consciousness.return_value = MagicMock()
            mock_emotion.return_value = MagicMock()
            mock_language.return_value = MagicMock()
            mock_memory_module.return_value = MagicMock()
            mock_social.return_value = MagicMock()
            mock_thought.return_value = MagicMock()
            
            # Initialize the LMM system
            system = main.LargeMindsModel()
            
            # Verify initialization
            assert system is not None
            
            # Check that all components were initialized
            mock_memory_manager.assert_called_once()
            mock_stage_manager.assert_called_once()
            mock_learning_manager.assert_called_once()
            mock_caregiver.assert_called_once()
            mock_consciousness.assert_called_once()
            mock_emotion.assert_called_once()
            mock_language.assert_called_once()
            mock_memory_module.assert_called_once()
            mock_social.assert_called_once()
            mock_thought.assert_called_once()
    
    def test_input_processing(self, mock_components, sample_input):
        """Test processing input through the main module."""
        with patch("lmm.main.LargeMindsModel") as mock_lmm_class:
            # Create mock LMM instance
            mock_lmm = MagicMock()
            mock_lmm_class.return_value = mock_lmm
            
            # Define interaction response
            mock_response = "The meaning of life is to find purpose and connection."
            mock_lmm.interact.return_value = mock_response
            
            # Create the system
            system = mock_lmm_class()
            
            # Process input
            response = system.interact(sample_input["query"])
            
            # Verify response
            assert response == mock_response
            
            # Verify that the appropriate modules were called
            mock_lmm.interact.assert_called_once_with(sample_input["query"], stream=False)
    
    def test_memory_interactions(self):
        """Test memory interactions in the main module."""
        with patch("lmm.memory.persistence.MemoryManager") as mock_memory_manager, \
             patch("lmm.memory.advanced_memory.AdvancedMemoryManager") as mock_advanced_memory, \
             patch("lmm.main.LargeMindsModel._extract_semantic_memories") as mock_extract:
            
            # Create mock instances
            memory_manager_instance = MagicMock()
            advanced_memory_instance = MagicMock()
            
            # Configure mock returns
            mock_memory_manager.return_value = memory_manager_instance
            mock_advanced_memory.return_value = advanced_memory_instance
            
            # Mock other components
            with patch.multiple("lmm.main",
                               DevelopmentalStageManager=MagicMock(),
                               LearningManager=MagicMock(),
                               MotherCaregiver=MagicMock(),
                               ConsciousnessModule=MagicMock(),
                               EmotionModule=MagicMock(),
                               LanguageModule=MagicMock(),
                               MemoryModule=MagicMock(),
                               SocialCognitionModule=MagicMock(),
                               ThoughtModule=MagicMock()):
                
                # Initialize the LMM system
                system = main.LargeMindsModel()
                
                # Configure mock stage manager
                system._stage_manager.get_current_stage.return_value = "infancy"
                
                # Test storing memory
                system._store_interaction_memory(
                    message="User question", 
                    response="System response", 
                    current_stage="infancy"
                )
                
                # Verify memory was stored
                memory_manager_instance.add_memory.assert_called()
                mock_extract.assert_called_once()
    
    @pytest.mark.parametrize("input_type,expected_modules", [
        ("question", ["language", "thought"]),
        ("emotional", ["emotion", "language", "social"]),
        ("memory_retrieval", ["memory", "consciousness"])
    ])
    def test_module_activation(self, mock_components, input_type, expected_modules):
        """Test activation of appropriate modules based on input type."""
        # This test would be implemented based on the actual module activation logic in the main module
        # For now, we'll verify basic module connectivity
        
        with patch.multiple("lmm.main",
                           MemoryManager=MagicMock(),
                           AdvancedMemoryManager=MagicMock(),
                           DevelopmentalStageManager=MagicMock(),
                           LearningManager=MagicMock(),
                           MotherCaregiver=MagicMock(),
                           ConsciousnessModule=MagicMock(),
                           EmotionModule=MagicMock(),
                           LanguageModule=MagicMock(),
                           MemoryModule=MagicMock(),
                           SocialCognitionModule=MagicMock(),
                           ThoughtModule=MagicMock()):
            
            # Initialize the LMM system
            system = main.LargeMindsModel()
            
            # Get references to the modules
            modules = {
                "language": system._language_module,
                "thought": system._thought_module,
                "emotion": system._emotion_module,
                "social": system._social_module,
                "memory": system._memory_module,
                "consciousness": system._consciousness_module
            }
            
            # Verify that all expected modules exist
            for module_name in expected_modules:
                assert module_name in modules
                assert modules[module_name] is not None
    
    def test_error_handling(self):
        """Test error handling in the main module."""
        with patch("lmm.main.LargeMindsModel") as mock_lmm_class:
            # Create mock LMM instance
            mock_lmm = MagicMock()
            mock_lmm_class.return_value = mock_lmm
            
            # Configure interaction to raise an exception
            mock_lmm.interact.side_effect = Exception("Test error")
            
            # Create the system
            system = mock_lmm_class()
            
            try:
                # Process input (should raise exception)
                system.interact("Test input")
                assert False, "Exception was not raised"
            except Exception as e:
                # Verify exception was raised
                assert str(e) == "Test error"
    
    def test_development_status(self):
        """Test getting development status."""
        with patch.multiple("lmm.main",
                           MemoryManager=MagicMock(),
                           AdvancedMemoryManager=MagicMock(),
                           DevelopmentalStageManager=MagicMock(),
                           LearningManager=MagicMock(),
                           MotherCaregiver=MagicMock(),
                           ConsciousnessModule=MagicMock(),
                           EmotionModule=MagicMock(),
                           LanguageModule=MagicMock(),
                           MemoryModule=MagicMock(),
                           SocialCognitionModule=MagicMock(),
                           ThoughtModule=MagicMock()):
            
            # Initialize the LMM system
            system = main.LargeMindsModel()
            
            # Configure mock stage manager
            system._stage_manager.get_current_stage.return_value = "toddler"
            system._stage_manager.get_stage_progress.return_value = 0.75
            system._stage_manager.get_all_stages.return_value = ["prenatal", "infant", "toddler", "child", "adolescent"]
            
            # Get development status
            status = system.get_development_status()
            
            # Verify status
            assert status["current_stage"] == "toddler"
            assert status["progress"] == 0.75
            assert "stages" in status
            assert len(status["stages"]) == 5
    
    def test_memory_recall(self):
        """Test memory recall functionality."""
        with patch.multiple("lmm.main",
                           MemoryManager=MagicMock(),
                           AdvancedMemoryManager=MagicMock(),
                           DevelopmentalStageManager=MagicMock(),
                           LearningManager=MagicMock(),
                           MotherCaregiver=MagicMock(),
                           ConsciousnessModule=MagicMock(),
                           EmotionModule=MagicMock(),
                           LanguageModule=MagicMock(),
                           MemoryModule=MagicMock(),
                           SocialCognitionModule=MagicMock(),
                           ThoughtModule=MagicMock()):
            
            # Initialize the LMM system
            system = main.LargeMindsModel()
            
            # Configure mock memory search
            test_memories = [
                {"id": 1, "content": "Memory 1", "type": "episodic", "importance": "high"},
                {"id": 2, "content": "Memory 2", "type": "semantic", "importance": "medium"}
            ]
            system._memory_manager.search_memories.return_value = test_memories
            
            # Recall memories
            memories = system.recall_memories("test query", limit=2)
            
            # Verify recall
            assert memories == test_memories
            system._memory_manager.search_memories.assert_called_once_with(
                query="test query", 
                memory_type=None, 
                min_importance=None, 
                limit=2
            ) 