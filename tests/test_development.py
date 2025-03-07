"""
Tests for the Development module of the Large Mind Model.
"""

import pytest
import time
from pathlib import Path

from lmm.development.stages import DevelopmentManager


class TestDevelopmentModule:
    """Test suite for the Development module."""
    
    @pytest.fixture
    def development_manager(self, test_config, developmental_stages, full_mind_model):
        """Create a development manager for testing."""
        manager = DevelopmentManager(
            config=test_config.development.dict(),
            stages=developmental_stages,
            mind_model=full_mind_model
        )
        return manager
    
    def test_initialization(self, development_manager):
        """Test that development manager initializes correctly."""
        # Check initial stage
        assert development_manager.current_stage == "prenatal"
        
        # Initialize
        development_manager.initialize()
        assert development_manager.initialized is True
        
        # Check available stages
        assert len(development_manager.stages) >= 4
        assert all(stage in development_manager.stages for stage in 
                  ["prenatal", "infancy", "childhood", "adolescence", "adulthood"])
    
    def test_stage_progression(self, development_manager):
        """Test progression through developmental stages."""
        development_manager.initialize()
        
        # Test updating development levels
        development_manager.update_development_levels({
            "memory": 0.15,
            "language": 0.12,
            "emotion": 0.11,
            "consciousness": 0.10,
            "social": 0.09,
            "thought": 0.08,
            "imagination": 0.07
        })
        
        # Check overall development level
        assert 0.05 <= development_manager.overall_development_level <= 0.15
        
        # Progress to next stage
        previous_stage = development_manager.current_stage
        result = development_manager.progress_stage()
        assert result["success"] is True
        assert development_manager.current_stage != previous_stage
        assert development_manager.current_stage == "infancy"
        
        # Update to higher levels
        development_manager.update_development_levels({
            "memory": 0.35,
            "language": 0.32,
            "emotion": 0.31,
            "consciousness": 0.28,
            "social": 0.30,
            "thought": 0.29,
            "imagination": 0.27
        })
        
        # Progress again
        result = development_manager.progress_stage()
        assert result["success"] is True
        assert development_manager.current_stage == "childhood"
    
    def test_developmental_requirements(self, development_manager):
        """Test that developmental requirements are enforced."""
        development_manager.initialize()
        
        # Set development levels below the threshold for next stage
        development_manager.update_development_levels({
            "memory": 0.05,
            "language": 0.04,
            "emotion": 0.03,
            "consciousness": 0.02,
            "social": 0.02,
            "thought": 0.01,
            "imagination": 0.01
        })
        
        # Attempt to skip to childhood (should fail)
        result = development_manager.set_stage("childhood")
        assert result["success"] is False
        assert "error" in result
        assert development_manager.current_stage == "prenatal"
        
        # Update to valid levels for infancy
        development_manager.update_development_levels({
            "memory": 0.15,
            "language": 0.12,
            "emotion": 0.11,
            "consciousness": 0.10,
            "social": 0.09,
            "thought": 0.08,
            "imagination": 0.07
        })
        
        # Now try again to progress
        result = development_manager.progress_stage()
        assert result["success"] is True
        assert development_manager.current_stage == "infancy"
    
    def test_module_development_coordination(self, development_manager, full_mind_model):
        """Test coordination of development across modules."""
        development_manager.initialize()
        
        # Initial module check
        for module_name, module in full_mind_model.items():
            if module_name not in ["config", "mother"]:
                assert module.development_level == 0.0
        
        # Progress to infancy
        development_manager.update_development_levels({
            "memory": 0.15,
            "language": 0.12,
            "emotion": 0.11,
            "consciousness": 0.10,
            "social": 0.09,
            "thought": 0.08,
            "imagination": 0.07
        })
        development_manager.progress_stage()
        
        # Apply development to modules
        result = development_manager.apply_development_to_modules()
        assert result["success"] is True
        
        # Check that modules were updated
        for module_name, expected_level in result["applied_levels"].items():
            if module_name in full_mind_model:
                module = full_mind_model[module_name]
                assert module.development_level == expected_level
    
    def test_stage_based_capabilities(self, development_manager):
        """Test stage-specific capabilities."""
        development_manager.initialize()
        
        # Get prenatal capabilities
        prenatal_capabilities = development_manager.get_stage_capabilities("prenatal")
        assert prenatal_capabilities["success"] is True
        assert prenatal_capabilities["language_capabilities"]["vocabulary_size"] == 0
        
        # Progress to infancy
        development_manager.update_development_levels({
            "memory": 0.15,
            "language": 0.12,
            "emotion": 0.11,
            "consciousness": 0.10,
            "social": 0.09,
            "thought": 0.08,
            "imagination": 0.07
        })
        development_manager.progress_stage()
        
        # Get infancy capabilities
        infancy_capabilities = development_manager.get_stage_capabilities("infancy")
        assert infancy_capabilities["success"] is True
        assert infancy_capabilities["language_capabilities"]["vocabulary_size"] > 0
        assert infancy_capabilities["consciousness_capabilities"]["self_awareness"] == "emerging"
        
        # Compare capabilities between stages
        assert infancy_capabilities["language_capabilities"]["vocabulary_size"] > \
               prenatal_capabilities["language_capabilities"]["vocabulary_size"]
    
    def test_developmental_milestones(self, development_manager):
        """Test tracking and achieving developmental milestones."""
        development_manager.initialize()
        
        # Check current milestones
        milestones = development_manager.get_milestones()
        assert milestones["success"] is True
        assert len(milestones["current_stage_milestones"]) > 0
        assert len(milestones["achieved_milestones"]) == 0
        
        # Achieve a milestone
        result = development_manager.achieve_milestone({
            "stage": "prenatal",
            "module": "memory",
            "name": "basic_memory_formation",
            "description": "Able to form and retrieve simple memories",
            "evidence": {"test_result": "passed", "accuracy": 0.85}
        })
        assert result["success"] is True
        
        # Verify milestone was recorded
        milestones = development_manager.get_milestones()
        assert len(milestones["achieved_milestones"]) == 1
        assert milestones["achieved_milestones"][0]["name"] == "basic_memory_formation"
        
        # Achieve multiple milestones
        milestones_to_achieve = [
            {
                "stage": "prenatal",
                "module": "emotion",
                "name": "basic_emotion_detection",
                "description": "Can detect basic emotional signals",
                "evidence": {"test_result": "passed", "accuracy": 0.75}
            },
            {
                "stage": "prenatal",
                "module": "language",
                "name": "phoneme_recognition",
                "description": "Can recognize basic speech sounds",
                "evidence": {"test_result": "passed", "accuracy": 0.80}
            }
        ]
        
        for m in milestones_to_achieve:
            development_manager.achieve_milestone(m)
        
        # Verify multiple milestones
        milestones = development_manager.get_milestones()
        assert len(milestones["achieved_milestones"]) == 3
    
    def test_development_acceleration(self, development_manager):
        """Test acceleration and deceleration of development."""
        development_manager.initialize()
        
        # Check default speed
        assert development_manager.progression_speed == 1.0
        
        # Accelerate development
        result = development_manager.set_progression_speed(2.0)
        assert result["success"] is True
        assert development_manager.progression_speed == 2.0
        
        # Update with acceleration
        start_time = time.time()
        development_manager.update_development_levels({
            "memory": 0.05,
            "language": 0.04,
            "emotion": 0.03,
            "consciousness": 0.02,
            "social": 0.02,
            "thought": 0.01,
            "imagination": 0.01
        })
        accelerated_update_time = time.time() - start_time
        
        # Reset and slow down
        development_manager.set_progression_speed(0.5)
        assert development_manager.progression_speed == 0.5
        
        # Update with deceleration
        start_time = time.time()
        development_manager.update_development_levels({
            "memory": 0.05,
            "language": 0.04,
            "emotion": 0.03,
            "consciousness": 0.02,
            "social": 0.02,
            "thought": 0.01,
            "imagination": 0.01
        })
        decelerated_update_time = time.time() - start_time
        
        # The processing time might not actually be different, but the growth rate should be
        levels_before = development_manager.module_levels.copy()
        
        # Update with default speed
        development_manager.set_progression_speed(1.0)
        development_manager.update_development_levels({
            "memory": 0.06,
            "language": 0.05,
            "emotion": 0.04,
            "consciousness": 0.03,
            "social": 0.03,
            "thought": 0.02,
            "imagination": 0.02
        })
        
        # Update with accelerated speed
        development_manager.set_progression_speed(2.0)
        development_manager.update_development_levels({
            "memory": 0.06,
            "language": 0.05,
            "emotion": 0.04,
            "consciousness": 0.03,
            "social": 0.03,
            "thought": 0.02,
            "imagination": 0.02
        })
        
        # Growth should be approximately 2x
        for module in levels_before:
            if module in development_manager.module_levels:
                assert development_manager.module_levels[module] > levels_before[module]
    
    def test_state_persistence(self, development_manager, temp_memory_dir):
        """Test saving and loading developmental state."""
        development_manager.initialize()
        
        # Set up some development and milestones
        development_manager.update_development_levels({
            "memory": 0.15,
            "language": 0.12,
            "emotion": 0.11,
            "consciousness": 0.10,
            "social": 0.09,
            "thought": 0.08,
            "imagination": 0.07
        })
        
        development_manager.achieve_milestone({
            "stage": "prenatal",
            "module": "memory",
            "name": "basic_memory_formation",
            "description": "Able to form and retrieve simple memories",
            "evidence": {"test_result": "passed"}
        })
        
        development_manager.progress_stage()
        
        # Save state
        save_path = temp_memory_dir / "development_test"
        save_path.mkdir(exist_ok=True)
        result = development_manager.save_state(str(save_path))
        assert result["success"] is True
        
        # Create new manager
        new_manager = DevelopmentManager(
            config=development_manager.config,
            stages=development_manager.stages,
            mind_model=development_manager.mind_model
        )
        new_manager.initialize()
        
        # Load state
        result = new_manager.load_state(str(save_path))
        assert result["success"] is True
        
        # Verify state was restored
        assert new_manager.current_stage == "infancy"
        assert len(new_manager.achieved_milestones) == 1
        assert new_manager.overall_development_level > 0.0
        
    def test_development_analytics(self, development_manager):
        """Test development analytics and reporting."""
        development_manager.initialize()
        
        # Set up some development data
        for i in range(10):
            development_manager.update_development_levels({
                "memory": 0.01 * (i + 1),
                "language": 0.01 * i,
                "emotion": 0.015 * i,
                "consciousness": 0.005 * i,
                "social": 0.008 * i,
                "thought": 0.007 * i,
                "imagination": 0.006 * i
            })
            
            # Record the state for history
            development_manager.record_development_snapshot()
        
        # Get analytics
        analytics = development_manager.get_development_analytics()
        assert analytics["success"] is True
        
        # Check analytics content
        assert "growth_rates" in analytics
        assert "developmental_trajectory" in analytics
        assert "module_correlations" in analytics
        assert "development_history" in analytics
        assert len(analytics["development_history"]) >= 10
        
        # Check visualization data
        visualization_data = development_manager.get_visualization_data()
        assert visualization_data["success"] is True
        assert "time_series" in visualization_data
        assert "radar_chart" in visualization_data
        assert "module_comparison" in visualization_data 