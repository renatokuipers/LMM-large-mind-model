"""
Pytest configuration and shared fixtures for the Large Mind Model.

This module provides fixtures and configurations that can be shared across multiple test modules.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import faiss

from lmm.config import Config, LLMConfig, MemoryConfig
from lmm.core.memory import MemoryModule
from lmm.core.consciousness import ConsciousnessModule
from lmm.core.language import LanguageModule
from lmm.core.emotion import EmotionModule
from lmm.core.social import SocialModule
from lmm.core.thought import ThoughtModule
from lmm.core.imagination import ImaginationModule
from lmm.mother.mother_llm import MotherLLM


@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration."""
    config = Config()
    config.llm.base_url = "http://localhost:8000"  # Mock server for testing
    config.memory.vector_dimensions = 512  # Smaller for testing
    config.debug_mode = True
    config.log_level = "DEBUG"
    return config


@pytest.fixture(scope="function")
def temp_memory_dir():
    """Create a temporary directory for memory storage during tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_embedding():
    """Create mock embeddings for testing."""
    def _create_embedding(n_vectors=5, dim=512):
        return np.random.random((n_vectors, dim)).astype(np.float32)
    return _create_embedding


@pytest.fixture(scope="function")
def mock_faiss_index():
    """Create a mock FAISS index for testing."""
    def _create_index(dim=512):
        return faiss.IndexFlatL2(dim)
    return _create_index


@pytest.fixture(scope="function")
def memory_module(test_config, temp_memory_dir):
    """Create a memory module instance for testing."""
    config = test_config
    config.memory.memory_store_path = temp_memory_dir
    memory = MemoryModule(name="memory", config=config.memory.dict())
    yield memory
    # Cleanup


@pytest.fixture(scope="function")
def consciousness_module(test_config):
    """Create a consciousness module instance for testing."""
    module = ConsciousnessModule(name="consciousness", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def language_module(test_config):
    """Create a language module instance for testing."""
    module = LanguageModule(name="language", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def emotion_module(test_config):
    """Create an emotion module instance for testing."""
    module = EmotionModule(name="emotion", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def social_module(test_config):
    """Create a social module instance for testing."""
    module = SocialModule(name="social", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def thought_module(test_config):
    """Create a thought module instance for testing."""
    module = ThoughtModule(name="thought", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def imagination_module(test_config):
    """Create an imagination module instance for testing."""
    module = ImaginationModule(name="imagination", config=test_config.dict())
    yield module
    # Cleanup


@pytest.fixture(scope="function")
def mother_llm(test_config):
    """Create a Mother LLM instance for testing."""
    mother = MotherLLM(config=test_config.mother.dict())
    yield mother
    # Cleanup


@pytest.fixture(scope="function")
def mock_llm_response():
    """Create mock LLM responses for testing."""
    def _create_response(content_type="text", sentiment=None):
        if content_type == "text":
            return "This is a simulated LLM response for testing purposes."
        elif content_type == "json":
            return {
                "response": "This is a structured response",
                "sentiment": sentiment or "neutral",
                "confidence": 0.85
            }
        elif content_type == "embedding":
            return np.random.random(512).astype(np.float32).tolist()
        else:
            return None
    return _create_response


@pytest.fixture(scope="function")
def full_mind_model(
    test_config, memory_module, consciousness_module, language_module,
    emotion_module, social_module, thought_module, imagination_module,
    mother_llm
):
    """Create a full mind model with all modules connected."""
    # Connect all modules
    memory_module.connect("consciousness", consciousness_module)
    memory_module.connect("language", language_module)
    
    consciousness_module.connect("memory", memory_module)
    consciousness_module.connect("thought", thought_module)
    
    language_module.connect("memory", memory_module)
    language_module.connect("emotion", emotion_module)
    
    emotion_module.connect("memory", memory_module)
    emotion_module.connect("social", social_module)
    
    social_module.connect("memory", memory_module)
    social_module.connect("emotion", emotion_module)
    
    thought_module.connect("memory", memory_module)
    thought_module.connect("consciousness", consciousness_module)
    thought_module.connect("imagination", imagination_module)
    
    imagination_module.connect("memory", memory_module)
    imagination_module.connect("thought", thought_module)
    
    # Assemble the full model
    model = {
        "memory": memory_module,
        "consciousness": consciousness_module,
        "language": language_module,
        "emotion": emotion_module,
        "social": social_module,
        "thought": thought_module,
        "imagination": imagination_module,
        "mother": mother_llm,
        "config": test_config
    }
    
    return model


@pytest.fixture(scope="function")
def developmental_stages():
    """Create mock developmental stages for testing."""
    return {
        "prenatal": {"duration": 1, "min_development_level": 0.0, "max_development_level": 0.1},
        "infancy": {"duration": 5, "min_development_level": 0.1, "max_development_level": 0.3},
        "childhood": {"duration": 10, "min_development_level": 0.3, "max_development_level": 0.6},
        "adolescence": {"duration": 15, "min_development_level": 0.6, "max_development_level": 0.8},
        "adulthood": {"duration": -1, "min_development_level": 0.8, "max_development_level": 1.0}
    }


@pytest.fixture(scope="function")
def mock_memory_entries():
    """Create mock memory entries for testing."""
    return [
        {
            "text": "My first interaction with Mother",
            "embedding": None,  # Will be filled by the memory module
            "timestamp": 1646006400,
            "memory_type": "episodic",
            "emotion": "curiosity",
            "importance": 0.8
        },
        {
            "text": "The color red is associated with warmth",
            "embedding": None,
            "timestamp": 1646092800,
            "memory_type": "semantic",
            "emotion": "neutral",
            "importance": 0.5
        },
        {
            "text": "I learned about counting numbers today",
            "embedding": None,
            "timestamp": 1646179200,
            "memory_type": "episodic",
            "emotion": "joy",
            "importance": 0.7
        },
        {
            "text": "2 + 2 equals 4",
            "embedding": None,
            "timestamp": 1646179200,
            "memory_type": "semantic",
            "emotion": "neutral",
            "importance": 0.6
        },
        {
            "text": "I had a dream about flying",
            "embedding": None,
            "timestamp": 1646265600,
            "memory_type": "dream",
            "emotion": "excitement",
            "importance": 0.9
        }
    ] 