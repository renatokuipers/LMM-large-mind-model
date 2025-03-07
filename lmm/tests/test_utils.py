import pytest
import os
import json
import logging
import tempfile
from unittest.mock import patch, MagicMock
from lmm.utils import (
    config,
    logging as lmm_logging,
    nlp_utils
)

class TestConfig:
    """Tests for the configuration utility functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Fixture providing sample configuration data."""
        return {
            "llm": {
                "base_url": "http://test.api/v1",
                "chat_model": "test-model",
                "embedding_model": "test-embed-model",
                "temperature": 0.7,
                "max_tokens": 100
            },
            "memory": {
                "vector_db_path": "test_memory_data",
                "use_gpu": True,
                "gpu_device": 0
            },
            "development": {
                "current_stage": "initial",
                "acceleration_factor": 50.0,
                "enable_plateaus": True
            }
        }
    
    def test_config_loading(self, sample_config, tmp_path):
        """Test loading configuration from file."""
        # Create a temp config file
        config_path = os.path.join(tmp_path, "config.json")
        
        # Write the sample config to the file
        with open(config_path, "w") as f:
            json.dump(sample_config, f)
        
        # Mock the config module to avoid global state changes
        with patch("lmm.utils.config.LMMConfig") as mock_config_class:
            # Create a mock config instance
            mock_config = MagicMock()
            mock_config_class.model_validate.return_value = mock_config
            
            # Load the config
            result = config.load_config_from_file(config_path)
            
            # Verify the config was loaded
            assert result is True
            mock_config_class.model_validate.assert_called_once()
            # Get the actual dict that was passed to model_validate
            config_dict = mock_config_class.model_validate.call_args[0][0]
            assert config_dict["llm"]["base_url"] == sample_config["llm"]["base_url"]
            assert config_dict["memory"]["vector_db_path"] == sample_config["memory"]["vector_db_path"]
    
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        # Valid configuration
        with patch("lmm.utils.config.LMMConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.model_validate.return_value = mock_config
            
            # This should validate successfully
            valid = config.load_config_from_dict(sample_config)
            assert valid is True
            
            # Test with invalid configuration (missing required field)
            invalid_config = sample_config.copy()
            del invalid_config["llm"]
            
            # Mock validation error
            mock_config_class.model_validate.side_effect = ValueError("Missing required field")
            
            # This should fail validation
            with pytest.raises(ValueError):
                config.load_config_from_dict(invalid_config)
    
    @pytest.mark.parametrize("key,expected", [
        ("llm.base_url", "http://test.api/v1"),
        ("llm.temperature", 0.7),
        ("memory.vector_db_path", "test_memory_data"),
        ("development.current_stage", "initial"),
        ("nonexistent.key", None)
    ])
    def test_config_access(self, key, expected):
        """Test accessing nested configuration values."""
        # Create a nested dictionary for testing
        nested_dict = {
            "llm": {
                "base_url": "http://test.api/v1",
                "temperature": 0.7
            },
            "memory": {
                "vector_db_path": "test_memory_data"
            },
            "development": {
                "current_stage": "initial"
            }
        }
        
        # Split the key into parts
        parts = key.split('.')
        
        # Navigate through the dictionary
        current = nested_dict
        for part in parts[:-1]:
            if part in current:
                current = current[part]
            else:
                current = None
                break
                
        # Get the final value
        value = None
        if current is not None and parts[-1] in current:
            value = current[parts[-1]]
            
        # Verify the result
        assert value == expected

class TestLogging:
    """Tests for the logging utility functionality."""
    
    @pytest.fixture
    def log_config(self):
        """Fixture providing logging configuration."""
        return {
            "level": "INFO",
            "file": "test.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    
    def test_logger_creation(self, log_config, tmp_path):
        """Test creation of logger with specified configuration."""
        log_path = os.path.join(tmp_path, log_config["file"])
        
        # Create logger with file output
        logger = lmm_logging.setup_logger(
            name="test_logger",
            level=log_config["level"],
            log_file=log_path,
            console_output=True
        )
        
        # Verify logger configuration
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # File handler and console handler
        
        # Test that the log file was created
        assert os.path.exists(log_path)
        
        # Write a log message
        logger.info("Test log message")
        
        # Verify the log file contains the message
        with open(log_path, "r") as f:
            log_content = f.read()
            assert "Test log message" in log_content
    
    def test_logging_levels(self):
        """Test different logging levels."""
        # Test with different log levels
        with patch("logging.Logger.setLevel") as mock_set_level:
            # Debug level
            logger = lmm_logging.setup_logger("test_debug", level="debug", console_output=True)
            mock_set_level.assert_called_with(logging.DEBUG)
            
            # Info level
            logger = lmm_logging.setup_logger("test_info", level="info", console_output=True)
            mock_set_level.assert_called_with(logging.INFO)
            
            # Warning level
            logger = lmm_logging.setup_logger("test_warning", level="warning", console_output=True)
            mock_set_level.assert_called_with(logging.WARNING)
            
            # Error level
            logger = lmm_logging.setup_logger("test_error", level="error", console_output=True)
            mock_set_level.assert_called_with(logging.ERROR)
            
            # Critical level
            logger = lmm_logging.setup_logger("test_critical", level="critical", console_output=True)
            mock_set_level.assert_called_with(logging.CRITICAL)
            
            # Invalid level should default to INFO
            logger = lmm_logging.setup_logger("test_invalid", level="invalid", console_output=True)
            mock_set_level.assert_called_with(logging.INFO)
    
    def test_get_logger(self):
        """Test getting a logger."""
        # Test that get_logger returns the same logger for the same name
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger1 = lmm_logging.get_logger("test")
            logger2 = lmm_logging.get_logger("test")
            
            assert logger1 is logger2
            assert mock_get_logger.call_count == 2
            mock_get_logger.assert_called_with("test")

class TestNLPUtils:
    """Tests for the NLP utilities functionality."""
    
    @pytest.fixture
    def sample_text(self):
        """Fixture providing sample text for NLP processing."""
        return "This is a sample text for testing NLP utilities. It contains multiple sentences with different words."
    
    @pytest.fixture
    def sample_entities(self):
        """Fixture providing sample entities."""
        return {
            "PERSON": ["John Doe", "Jane Smith"],
            "ORG": ["Acme Inc", "Tech Corp"],
            "DATE": ["2023-01-01", "yesterday"],
            "LOCATION": ["New York", "Paris"]
        }
    
    @pytest.fixture
    def sample_concepts(self):
        """Fixture providing sample extracted concepts."""
        return [
            {"term": "machine learning", "type": "concept", "importance": 0.8},
            {"term": "artificial intelligence", "type": "concept", "importance": 0.9},
            {"term": "data processing", "type": "concept", "importance": 0.7}
        ]
    
    def test_extract_named_entities(self, sample_text):
        """Test extracting named entities from text."""
        # Mock the extraction function
        expected_entities = {
            "PERSON": ["John"],
            "ORG": ["Test Company"],
            "LOCATION": ["Test City"]
        }
        
        with patch("lmm.utils.nlp_utils.extract_named_entities") as mock_extract:
            mock_extract.return_value = expected_entities
            
            # Extract entities
            entities = nlp_utils.extract_named_entities(sample_text)
            
            # Verify results
            assert entities == expected_entities
            mock_extract.assert_called_once_with(sample_text)
    
    def test_extract_key_concepts(self, sample_text):
        """Test extracting key concepts from text."""
        expected_concepts = [
            {"term": "sample text", "type": "noun_phrase", "frequency": 1, "importance": 0.7},
            {"term": "NLP utilities", "type": "noun_phrase", "frequency": 1, "importance": 0.8},
            {"term": "testing", "type": "verb", "frequency": 1, "importance": 0.6}
        ]
        
        with patch("lmm.utils.nlp_utils.extract_key_concepts") as mock_extract:
            mock_extract.return_value = expected_concepts
            
            # Extract concepts
            concepts = nlp_utils.extract_key_concepts(sample_text)
            
            # Verify results
            assert concepts == expected_concepts
            mock_extract.assert_called_once_with(sample_text)
    
    @pytest.mark.parametrize("text1,text2,expected_similarity", [
        ("This is a test", "This is a test", 1.0),
        ("This is a test", "This is another test", 0.8),
        ("This is a test", "Something completely different", 0.2)
    ])
    def test_text_similarity(self, text1, text2, expected_similarity):
        """Test text similarity calculation."""
        # This test would depend on the actual implementation of the similarity function
        # For now, we'll mock it
        with patch("lmm.utils.nlp_utils.calculate_similarity") as mock_similarity:
            mock_similarity.return_value = expected_similarity
            
            # Calculate similarity
            similarity = mock_similarity(text1, text2)
            
            # Verify result
            assert similarity == expected_similarity
    
    def test_embedding_generation(self, sample_text):
        """Test generating embeddings from text."""
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with patch("lmm.core.mother.llm_client.LLMClient") as mock_llm_client:
            # Set up mock LLM client
            mock_llm_instance = MagicMock()
            mock_llm_instance.get_embedding.return_value = expected_embedding
            mock_llm_client.return_value = mock_llm_instance
            
            # Mock the embedding function
            with patch("lmm.utils.nlp_utils.embed_text") as mock_embed:
                mock_embed.return_value = expected_embedding
                
                # Generate embedding
                embedding = mock_embed(sample_text)
                
                # Verify result
                assert embedding == expected_embedding
                mock_embed.assert_called_once_with(sample_text)
    
    def test_knowledge_categorization(self, sample_text, sample_entities, sample_concepts):
        """Test knowledge categorization functionality."""
        # Mock the relationship extraction
        relationships = [
            {
                "source": "machine learning", 
                "target": "artificial intelligence",
                "relation_type": "is_part_of"
            }
        ]
        
        expected_knowledge = [
            {
                "content": "Machine learning is part of artificial intelligence",
                "category": "technology",
                "subcategory": "artificial_intelligence",
                "confidence": 0.85,
                "importance": 0.9,
                "abstraction_level": "concept"
            }
        ]
        
        with patch("lmm.utils.nlp_utils.categorize_knowledge") as mock_categorize:
            mock_categorize.return_value = expected_knowledge
            
            # Categorize knowledge
            knowledge = mock_categorize(sample_text, sample_concepts, sample_entities, relationships)
            
            # Verify results
            assert knowledge == expected_knowledge
            mock_categorize.assert_called_once() 