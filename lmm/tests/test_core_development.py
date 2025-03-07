import pytest
import os
from unittest.mock import patch, MagicMock
from lmm.core.development import (
    learning,
    stages,
    advanced_learning
)

class TestLearning:
    """Tests for the learning module functionality."""
    
    @pytest.fixture
    def learning_data(self):
        """Fixture providing sample learning data."""
        return {
            "input_data": "Sample learning input",
            "learning_rate": 0.01,
            "target_outcome": "desired result"
        }
    
    @pytest.fixture
    def learning_manager(self):
        """Fixture providing a learning manager instance."""
        with patch("lmm.core.development.learning.LearningManager.initialize"):
            manager = learning.LearningManager()
            manager._learning_rate = 0.05
            manager._learning_history = []
            manager._current_skill_levels = {
                "language": 0.2,
                "reasoning": 0.1,
                "memory": 0.3,
                "social": 0.1,
                "perception": 0.2
            }
            return manager
    
    def test_learning_process(self, learning_manager, learning_data):
        """Test the basic learning process."""
        # Process a learning experience
        result = learning_manager.learn(
            skill="language",
            experience="Learning new words and concepts",
            difficulty=0.5,
            success_rate=0.7
        )
        
        # Verify learning occurred
        assert result is not None
        assert "skill" in result
        assert result["skill"] == "language"
        assert "improvement" in result
        assert result["improvement"] > 0
        
        # Verify skill level increased
        assert learning_manager._current_skill_levels["language"] > 0.2
        
        # Verify learning history was updated
        assert len(learning_manager._learning_history) == 1
        assert learning_manager._learning_history[0]["skill"] == "language"
    
    @pytest.mark.parametrize("learning_rate,epochs,expected_improvement", [
        (0.01, 10, "minimal"),
        (0.05, 10, "moderate"),
        (0.1, 10, "significant")
    ])
    def test_learning_rates(self, learning_rate, epochs, expected_improvement):
        """Test different learning rates and their impact."""
        with patch("lmm.core.development.learning.LearningManager.initialize"):
            # Create manager with specified learning rate
            manager = learning.LearningManager()
            manager._learning_rate = learning_rate
            manager._current_skill_levels = {"test_skill": 0.5}
            
            # Simulate learning over multiple epochs
            total_improvement = 0
            for _ in range(epochs):
                result = manager.learn(
                    skill="test_skill",
                    experience="Practice session",
                    difficulty=0.5,
                    success_rate=0.8
                )
                total_improvement += result["improvement"]
            
            # Verify improvement based on learning rate
            if expected_improvement == "minimal":
                assert total_improvement < 0.1
            elif expected_improvement == "moderate":
                assert 0.1 <= total_improvement < 0.2
            else:  # significant
                assert total_improvement >= 0.2
    
    def test_skill_level_bounds(self, learning_manager):
        """Test that skill levels are bounded between 0 and 1."""
        # Try to learn beyond max level
        learning_manager._current_skill_levels["language"] = 0.95
        
        result = learning_manager.learn(
            skill="language",
            experience="Advanced concept mastery",
            difficulty=0.9,
            success_rate=0.9
        )
        
        # Verify the skill level is capped at 1.0
        assert learning_manager._current_skill_levels["language"] <= 1.0
        
        # Try a new skill (should start at 0)
        result = learning_manager.learn(
            skill="new_skill",
            experience="First experience",
            difficulty=0.3,
            success_rate=0.5
        )
        
        # Verify the new skill was added
        assert "new_skill" in learning_manager._current_skill_levels
        assert learning_manager._current_skill_levels["new_skill"] > 0

class TestDevelopmentStages:
    """Tests for the development stages functionality."""
    
    @pytest.fixture
    def stage_data(self):
        """Fixture providing sample stage data."""
        return {
            "current_stage": "initial",
            "progress": 0.3,
            "requirements": {
                "initial": {"min_learning": 0.2},
                "intermediate": {"min_learning": 0.5},
                "advanced": {"min_learning": 0.8}
            }
        }
    
    @pytest.fixture
    def stage_manager(self):
        """Fixture providing a stage manager instance."""
        with patch("lmm.core.development.stages.DevelopmentalStageManager.initialize"):
            manager = stages.DevelopmentalStageManager()
            manager._current_stage = "infant"
            manager._stage_progress = 0.4
            manager._developmental_stages = [
                "prenatal", "infant", "toddler", "child", "adolescent", "adult"
            ]
            manager._stage_requirements = {
                "infant": {"language": 0.2, "social": 0.1},
                "toddler": {"language": 0.4, "social": 0.3, "reasoning": 0.2},
                "child": {"language": 0.6, "social": 0.5, "reasoning": 0.4, "memory": 0.5}
            }
            manager._skill_levels = {
                "language": 0.3,
                "social": 0.2,
                "reasoning": 0.1,
                "memory": 0.3
            }
            return manager
    
    def test_stage_progression(self, stage_manager):
        """Test progression through development stages."""
        # Check current stage
        assert stage_manager._current_stage == "infant"
        assert stage_manager._stage_progress == 0.4
        
        # Update skill levels to meet toddler requirements
        stage_manager._skill_levels["language"] = 0.5  # Above 0.4 required
        stage_manager._skill_levels["social"] = 0.4    # Above 0.3 required
        stage_manager._skill_levels["reasoning"] = 0.3  # Above 0.2 required
        
        # Check for stage progression
        progressed = stage_manager.check_stage_progression()
        
        # Verify stage progressed
        assert progressed is True
        assert stage_manager._current_stage == "toddler"
        assert stage_manager._stage_progress == 0.0  # Should reset to 0 for new stage
    
    @pytest.mark.parametrize("progress,expected_stage", [
        (0.1, "initial"),
        (0.4, "initial"),
        (0.6, "intermediate"),
        (0.9, "advanced")
    ])
    def test_stage_determination(self, progress, expected_stage):
        """Test determination of developmental stage based on progress."""
        with patch("lmm.core.development.stages.DevelopmentalStageManager.initialize"):
            # Create manager with simple developmental stages
            manager = stages.DevelopmentalStageManager()
            manager._developmental_stages = ["initial", "intermediate", "advanced"]
            manager._stage_thresholds = {
                "initial": 0.0,
                "intermediate": 0.5,
                "advanced": 0.8
            }
            
            # Set progress
            manager._overall_progress = progress
            
            # Determine appropriate stage
            appropriate_stage = manager._determine_stage_from_progress(progress)
            
            # Verify stage determination
            assert appropriate_stage == expected_stage
    
    def test_get_current_stage(self, stage_manager):
        """Test getting the current developmental stage."""
        # Get current stage
        current_stage = stage_manager.get_current_stage()
        
        # Verify correct stage is returned
        assert current_stage == "infant"
        
        # Get progress
        progress = stage_manager.get_stage_progress()
        
        # Verify correct progress is returned
        assert progress == 0.4
    
    def test_set_stage(self, stage_manager):
        """Test manually setting the developmental stage."""
        # Set stage to child
        stage_manager.set_stage("child")
        
        # Verify stage was set
        assert stage_manager._current_stage == "child"
        assert stage_manager._stage_progress == 0.0  # Progress should reset
        
        # Test setting invalid stage
        with pytest.raises(ValueError):
            stage_manager.set_stage("nonexistent_stage")

class TestAdvancedLearning:
    """Tests for the advanced learning functionality."""
    
    @pytest.fixture
    def advanced_learning_data(self):
        """Fixture providing advanced learning test data."""
        return {
            "input_sequence": ["data1", "data2", "data3"],
            "learning_algorithm": "reinforcement",
            "hyperparameters": {
                "discount_factor": 0.9,
                "exploration_rate": 0.2
            }
        }
    
    @pytest.fixture
    def advanced_learning_manager(self):
        """Fixture providing an advanced learning manager instance."""
        with patch("lmm.core.development.advanced_learning.AdvancedLearningManager.initialize"):
            manager = advanced_learning.AdvancedLearningManager()
            manager._learning_algorithms = {
                "supervised": MagicMock(),
                "reinforcement": MagicMock(),
                "unsupervised": MagicMock()
            }
            manager._current_model = None
            manager._learning_history = []
            return manager
    
    def test_advanced_learning_process(self, advanced_learning_manager, advanced_learning_data):
        """Test advanced learning processes."""
        # Configure mock learning algorithm
        advanced_learning_manager._learning_algorithms["reinforcement"].train.return_value = {
            "model": MagicMock(),
            "metrics": {
                "accuracy": 0.85,
                "loss": 0.2
            }
        }
        
        # Perform advanced learning
        result = advanced_learning_manager.learn(
            data=advanced_learning_data["input_sequence"],
            algorithm=advanced_learning_data["learning_algorithm"],
            hyperparameters=advanced_learning_data["hyperparameters"]
        )
        
        # Verify learning occurred
        assert result is not None
        assert "model" in result
        assert "metrics" in result
        assert result["metrics"]["accuracy"] == 0.85
        
        # Verify the learning algorithm was called with correct parameters
        advanced_learning_manager._learning_algorithms["reinforcement"].train.assert_called_once()
        
        # Verify learning history was updated
        assert len(advanced_learning_manager._learning_history) == 1
        assert advanced_learning_manager._learning_history[0]["algorithm"] == "reinforcement"
    
    @pytest.mark.parametrize("algorithm,expected_convergence", [
        ("supervised", "fast"),
        ("reinforcement", "moderate"),
        ("unsupervised", "slow")
    ])
    def test_learning_algorithms(self, advanced_learning_manager, algorithm, expected_convergence):
        """Test different learning algorithms and their convergence characteristics."""
        # Configure mock learning algorithms with different convergence rates
        convergence_rates = {
            "fast": 10,
            "moderate": 50,
            "slow": 100
        }
        
        advanced_learning_manager._learning_algorithms[algorithm].get_expected_convergence_iterations.return_value = (
            convergence_rates[expected_convergence]
        )
        
        # Get expected convergence
        iterations = advanced_learning_manager.get_expected_convergence(algorithm)
        
        # Verify convergence rate matches expectation
        assert iterations == convergence_rates[expected_convergence]
    
    def test_algorithm_selection(self, advanced_learning_manager):
        """Test automatic selection of appropriate learning algorithm."""
        # Configure mock task analyzer to recommend reinforcement learning
        with patch("lmm.core.development.advanced_learning.AdvancedLearningManager._analyze_task") as mock_analyze:
            mock_analyze.return_value = "reinforcement"
            
            # Select algorithm for a task
            algorithm = advanced_learning_manager.select_algorithm(
                task_type="sequential_decision",
                data_characteristics={
                    "sequential": True,
                    "labeled": False,
                    "sparse_rewards": True
                }
            )
            
            # Verify correct algorithm was selected
            assert algorithm == "reinforcement"
            mock_analyze.assert_called_once() 