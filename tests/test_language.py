"""
Tests for the Language module of the Large Mind Model.
"""

import pytest
import numpy as np
import time
from pathlib import Path

from lmm.core.language import LanguageModule


class TestLanguageModule:
    """Test suite for the Language module."""
    
    def test_language_initialization(self, language_module):
        """Test that language module initializes correctly."""
        assert language_module.name == "language"
        assert not language_module.initialized
        
        # Initialize the module
        result = language_module.initialize()
        assert result is True
        assert language_module.initialized
        assert language_module.development_level == 0.0
    
    def test_vocabulary_learning(self, language_module):
        """Test vocabulary acquisition."""
        language_module.initialize()
        
        # Learn a simple word
        result = language_module.process({
            "action": "learn_word",
            "word": "apple",
            "definition": "A round fruit with red, yellow, or green skin and firm white flesh.",
            "examples": ["I ate an apple for lunch.", "The apple tree is blooming."],
            "associations": ["fruit", "red", "sweet"],
            "importance": 0.7
        })
        assert result["success"] is True
        
        # Learn another word
        result = language_module.process({
            "action": "learn_word",
            "word": "happy",
            "definition": "Feeling or showing pleasure or contentment.",
            "examples": ["I am happy to see you.", "The happy child played in the garden."],
            "associations": ["joy", "smile", "pleasure"],
            "part_of_speech": "adjective",
            "importance": 0.8
        })
        assert result["success"] is True
        
        # Check vocabulary size
        result = language_module.process({"action": "get_vocabulary_stats"})
        assert result["success"] is True
        assert result["vocabulary_size"] >= 2
        assert "apple" in result["recently_learned"]
        assert "happy" in result["recently_learned"]
    
    def test_word_retrieval(self, language_module):
        """Test word retrieval from vocabulary."""
        language_module.initialize()
        
        # Add words to vocabulary
        words = [
            {"word": "dog", "definition": "A domesticated carnivorous mammal.", "part_of_speech": "noun"},
            {"word": "run", "definition": "Move at a speed faster than a walk.", "part_of_speech": "verb"},
            {"word": "blue", "definition": "A color like that of the sky on a clear day.", "part_of_speech": "adjective"},
            {"word": "quickly", "definition": "At a fast speed; rapidly.", "part_of_speech": "adverb"}
        ]
        
        for word_data in words:
            language_module.process({
                "action": "learn_word",
                **word_data
            })
        
        # Retrieve specific word
        result = language_module.process({
            "action": "retrieve_word",
            "word": "dog"
        })
        assert result["success"] is True
        assert result["word_data"]["word"] == "dog"
        assert "definition" in result["word_data"]
        
        # Retrieve words by part of speech
        result = language_module.process({
            "action": "retrieve_words_by_pos",
            "part_of_speech": "adjective"
        })
        assert result["success"] is True
        assert any(w["word"] == "blue" for w in result["words"])
        
        # Retrieve words by semantic association
        result = language_module.process({
            "action": "retrieve_words_by_association",
            "association": "animal"
        })
        assert result["success"] is True
        assert any(w["word"] == "dog" for w in result["words"])
    
    def test_sentence_understanding(self, language_module):
        """Test sentence parsing and understanding."""
        language_module.initialize()
        
        # Parse a simple sentence
        result = language_module.process({
            "action": "parse_sentence",
            "sentence": "The quick brown fox jumps over the lazy dog."
        })
        assert result["success"] is True
        assert "subject" in result["parsing"]
        assert "predicate" in result["parsing"]
        assert "tokens" in result["parsing"]
        
        # Understand a sentence
        result = language_module.process({
            "action": "understand_sentence",
            "sentence": "The child ate an apple because she was hungry."
        })
        assert result["success"] is True
        assert "entities" in result["understanding"]
        assert "relationships" in result["understanding"]
        assert "cause_effect" in result["understanding"]
        
        # Test more complex understanding
        result = language_module.process({
            "action": "understand_sentence",
            "sentence": "After finishing homework, John went to the park to meet his friends."
        })
        assert result["success"] is True
        assert "temporal_sequence" in result["understanding"]
        assert "purpose" in result["understanding"]
    
    def test_grammar_learning(self, language_module):
        """Test grammar rule learning and application."""
        language_module.initialize()
        
        # Learn a grammar rule
        result = language_module.process({
            "action": "learn_grammar_rule",
            "rule_name": "subject_verb_agreement",
            "description": "Subjects and verbs must agree in number (singular or plural).",
            "examples": [
                {"correct": "The dog barks.", "explanation": "Singular subject with singular verb"},
                {"correct": "The dogs bark.", "explanation": "Plural subject with plural verb"},
                {"incorrect": "The dog bark.", "explanation": "Singular subject with plural verb"}
            ]
        })
        assert result["success"] is True
        
        # Apply grammar rule to check sentence
        result = language_module.process({
            "action": "check_grammar",
            "sentence": "The cat sleeps on the mat.",
            "rules": ["subject_verb_agreement"]
        })
        assert result["success"] is True
        assert result["is_correct"] is True
        
        # Apply grammar rule to incorrect sentence
        result = language_module.process({
            "action": "check_grammar",
            "sentence": "The cats sleeps on the mat.",
            "rules": ["subject_verb_agreement"]
        })
        assert result["success"] is True
        assert result["is_correct"] is False
        assert "corrections" in result
    
    def test_language_generation(self, language_module):
        """Test language generation capabilities."""
        language_module.initialize()
        
        # Generate a simple sentence
        result = language_module.process({
            "action": "generate_sentence",
            "subject": "The cat",
            "complexity": "simple"
        })
        assert result["success"] is True
        assert "sentence" in result
        assert result["sentence"].startswith("The cat")
        
        # Generate a sentence with specified semantics
        result = language_module.process({
            "action": "generate_sentence",
            "semantics": {
                "action": "eat",
                "actor": "child",
                "object": "apple",
                "time": "morning"
            }
        })
        assert result["success"] is True
        assert "sentence" in result
        assert "child" in result["sentence"].lower()
        assert "eat" in result["sentence"].lower() or "ate" in result["sentence"].lower()
        assert "apple" in result["sentence"].lower()
        
        # Generate a more complex sentence
        result = language_module.process({
            "action": "generate_sentence",
            "complexity": "complex",
            "include_elements": ["temporal_clause", "purpose_clause"]
        })
        assert result["success"] is True
        assert "sentence" in result
        # Should contain elements like "when", "because", "in order to", etc.
    
    def test_conversation_skills(self, language_module, mother_llm):
        """Test conversational capabilities."""
        language_module.initialize()
        
        # Process a simple greeting
        result = language_module.process({
            "action": "process_utterance",
            "text": "Hello, how are you?",
            "speaker": "mother"
        })
        assert result["success"] is True
        assert "understanding" in result
        assert "appropriate_responses" in result
        
        # Generate a response
        result = language_module.process({
            "action": "generate_response",
            "to_utterance": "What is your favorite color?",
            "speaker": "mother",
            "context": {"previous_topics": ["colors", "preferences"]}
        })
        assert result["success"] is True
        assert "response" in result
        assert len(result["response"]) > 0
        
        # Have a multi-turn conversation
        conversation = [
            {"speaker": "mother", "text": "Would you like to learn about animals?"},
            {"speaker": "self", "text": "Yes, I would like that."},
            {"speaker": "mother", "text": "Great! Let's start with dogs. Dogs are loyal animals that people keep as pets."}
        ]
        
        result = language_module.process({
            "action": "process_conversation",
            "conversation": conversation
        })
        assert result["success"] is True
        assert "topic" in result
        assert result["topic"] == "animals"
        assert "learning_opportunities" in result
        
        # Generate next response in conversation
        result = language_module.process({
            "action": "continue_conversation",
            "conversation": conversation
        })
        assert result["success"] is True
        assert "next_utterance" in result
        assert len(result["next_utterance"]) > 0
    
    def test_linguistic_development(self, language_module):
        """Test linguistic development through stages."""
        language_module.initialize()
        
        # Check initial stage
        result = language_module.process({"action": "get_linguistic_stage"})
        assert result["success"] is True
        assert result["stage"] == "pre_linguistic"
        
        # Develop to early word stage
        language_module.development_level = 0.2
        language_module.update({"linguistic_exposure": 500})  # Simulate exposure to language
        
        result = language_module.process({"action": "get_linguistic_stage"})
        assert result["stage"] == "single_word"
        
        # Test single word expression
        result = language_module.process({
            "action": "express",
            "meaning": "want apple"
        })
        assert result["success"] is True
        assert result["expression"] == "Apple" or result["expression"] == "Want"
        
        # Develop to two-word stage
        language_module.development_level = 0.35
        language_module.update({"linguistic_exposure": 1000})
        
        result = language_module.process({"action": "get_linguistic_stage"})
        assert result["stage"] == "two_word"
        
        # Test two-word expression
        result = language_module.process({
            "action": "express",
            "meaning": "want apple"
        })
        assert result["success"] is True
        assert result["expression"] == "Want apple"
        
        # Develop to sentence stage
        language_module.development_level = 0.6
        language_module.update({"linguistic_exposure": 2000})
        
        result = language_module.process({"action": "get_linguistic_stage"})
        assert result["stage"] == "simple_sentences"
        
        # Test sentence expression
        result = language_module.process({
            "action": "express",
            "meaning": "child wants the red apple"
        })
        assert result["success"] is True
        assert "child" in result["expression"].lower()
        assert "want" in result["expression"].lower()
        assert "apple" in result["expression"].lower()
        
        # Develop to complex language
        language_module.development_level = 0.9
        language_module.update({"linguistic_exposure": 5000})
        
        result = language_module.process({"action": "get_linguistic_stage"})
        assert result["stage"] == "complex_language"
    
    def test_language_comprehension(self, language_module, memory_module):
        """Test language comprehension with memory integration."""
        language_module.initialize()
        memory_module.initialize()
        
        # Connect modules
        language_module.connect("memory", memory_module)
        
        # Learn some words first
        words = ["cat", "dog", "run", "sleep", "happy", "sad", "big", "small"]
        for word in words:
            language_module.process({
                "action": "learn_word",
                "word": word,
                "definition": f"Definition of {word}"
            })
        
        # Set development level to understand sentences
        language_module.development_level = 0.7
        
        # Comprehend simple text
        result = language_module.process({
            "action": "comprehend",
            "text": "The big dog runs. The small cat sleeps."
        })
        assert result["success"] is True
        assert "sentences" in result["comprehension"]
        assert "entities" in result["comprehension"]
        assert "big dog" in str(result["comprehension"]["entities"])
        assert "small cat" in str(result["comprehension"]["entities"])
        
        # Store comprehension in memory
        memory_id = memory_module.process({
            "action": "store",
            "memory": {
                "text": "The big dog runs. The small cat sleeps.",
                "embedding": None,  # Will be generated
                "timestamp": int(time.time()),
                "memory_type": "linguistic",
                "comprehension": result["comprehension"]
            }
        })["memory_id"]
        
        # Test recalling and explaining
        result = language_module.process({
            "action": "explain",
            "text": "Why does the cat sleep?",
            "context_memory_id": memory_id
        })
        assert result["success"] is True
        assert "explanation" in result 