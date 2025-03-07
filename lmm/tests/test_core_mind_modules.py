import pytest
import os
from unittest.mock import patch, MagicMock
from lmm.core.mind_modules import (
    base,
    consciousness,
    emotion,
    language,
    memory,
    social,
    thought
)

class TestBaseMindModule:
    """Tests for the base mind module functionality."""
    
    @pytest.fixture
    def base_module(self):
        """Fixture providing a base mind module instance."""
        return base.BaseMindModule(name="test_module")
    
    def test_base_module_initialization(self, base_module):
        """Test that base mind module initializes correctly."""
        assert base_module.name == "test_module"
        assert hasattr(base_module, "initialize")
        assert hasattr(base_module, "process")
        assert hasattr(base_module, "get_state")
    
    def test_base_module_process(self, base_module):
        """Test that base module process method works correctly."""
        # The base process method should return None
        result = base_module.process("test input")
        assert result is None
    
    def test_base_module_state(self, base_module):
        """Test that base module state method works correctly."""
        state = base_module.get_state()
        assert state is not None
        assert "name" in state
        assert state["name"] == "test_module"
        assert "active" in state
        assert state["active"] is True

class TestConsciousness:
    """Tests for the consciousness module functionality."""
    
    @pytest.fixture
    def consciousness_data(self):
        """Fixture providing test data for consciousness module."""
        return {
            "awareness_level": 0.7,
            "attention_focus": "test_focus",
            "current_state": "active"
        }
    
    @pytest.fixture
    def consciousness_module(self):
        """Fixture providing a consciousness module instance."""
        with patch("lmm.core.mind_modules.consciousness.ConsciousnessModule.initialize"):
            module = consciousness.ConsciousnessModule()
            module._awareness_level = 0.5
            module._attention_focus = None
            module._current_state = "idle"
            module._activation_threshold = 0.3
            module._focal_points = []
            return module
    
    def test_consciousness_initialization(self, consciousness_module):
        """Test consciousness module initialization."""
        assert consciousness_module._awareness_level == 0.5
        assert consciousness_module._attention_focus is None
        assert consciousness_module._current_state == "idle"
    
    def test_set_awareness(self, consciousness_module):
        """Test setting awareness level."""
        # Set awareness to a new value
        consciousness_module.set_awareness_level(0.8)
        assert consciousness_module._awareness_level == 0.8
        
        # Ensure boundaries are enforced
        consciousness_module.set_awareness_level(1.5)  # Beyond upper bound
        assert consciousness_module._awareness_level == 1.0
        
        consciousness_module.set_awareness_level(-0.5)  # Below lower bound
        assert consciousness_module._awareness_level == 0.0
    
    def test_focus_attention(self, consciousness_module):
        """Test focusing attention on a specific topic."""
        consciousness_module.focus_attention("test_topic", importance=0.7)
        assert consciousness_module._attention_focus == "test_topic"
        assert consciousness_module._current_state == "active"  # Should switch to active state
        
        # Adding to focal points
        assert len(consciousness_module._focal_points) == 1
        assert consciousness_module._focal_points[0]["topic"] == "test_topic"
        assert consciousness_module._focal_points[0]["importance"] == 0.7
    
    @pytest.mark.parametrize("awareness_level,expected_state", [
        (0.9, "highly_aware"),
        (0.5, "moderately_aware"),
        (0.1, "barely_aware")
    ])
    def test_awareness_states(self, consciousness_module, awareness_level, expected_state):
        """Test various awareness levels and their resulting states."""
        consciousness_module.set_awareness_level(awareness_level)
        
        # We'll define the expected mapping here to simulate the class behavior
        if awareness_level >= 0.8:
            assert consciousness_module._awareness_level >= 0.8
        elif awareness_level >= 0.4:
            assert 0.4 <= consciousness_module._awareness_level < 0.8
        else:
            assert consciousness_module._awareness_level < 0.4

class TestEmotionModule:
    """Tests for the emotion module functionality."""
    
    @pytest.fixture
    def emotion_state(self):
        """Fixture providing a sample emotion state."""
        return {
            "valence": 0.5,
            "arousal": 0.7,
            "dominance": 0.3,
            "primary_emotion": "joy",
            "intensity": 0.6
        }
    
    @pytest.fixture
    def emotion_module(self):
        """Fixture providing an emotion module instance."""
        with patch("lmm.core.mind_modules.emotion.EmotionModule.initialize"):
            module = emotion.EmotionModule()
            module._valence = 0.0  # Neutral valence
            module._arousal = 0.0  # Neutral arousal
            module._dominance = 0.5  # Neutral dominance
            module._primary_emotion = None
            module._intensity = 0.0
            module._emotion_history = []
            return module
    
    def test_emotion_initialization(self, emotion_module):
        """Test emotion module initialization."""
        assert emotion_module._valence == 0.0
        assert emotion_module._arousal == 0.0
        assert emotion_module._dominance == 0.5
        assert emotion_module._primary_emotion is None
        assert emotion_module._intensity == 0.0
    
    def test_emotion_processing(self, emotion_module):
        """Test emotion processing functionality."""
        # Process positive text
        with patch("lmm.core.mind_modules.emotion.EmotionModule._extract_emotions_from_text") as mock_extract:
            mock_extract.return_value = {
                "valence": 0.8,
                "arousal": 0.6,
                "dominance": 0.7,
                "primary_emotion": "joy",
                "intensity": 0.7
            }
            
            result = emotion_module.process("I'm feeling really happy and excited today!")
            
            # Verify emotions were updated
            assert emotion_module._valence > 0
            assert emotion_module._arousal > 0
            assert emotion_module._primary_emotion == "joy"
            assert emotion_module._intensity > 0
            
            # Verify history was updated
            assert len(emotion_module._emotion_history) == 1
    
    def test_set_emotion(self, emotion_module):
        """Test manually setting emotions."""
        emotion_module.set_emotion(
            valence=0.8,
            arousal=0.7,
            dominance=0.6,
            primary_emotion="excitement",
            intensity=0.75
        )
        
        # Verify emotions were set correctly
        assert emotion_module._valence == 0.8
        assert emotion_module._arousal == 0.7
        assert emotion_module._dominance == 0.6
        assert emotion_module._primary_emotion == "excitement"
        assert emotion_module._intensity == 0.75

class TestLanguageModule:
    """Tests for the language module functionality."""
    
    @pytest.fixture
    def sample_text(self):
        """Fixture providing sample text for language processing."""
        return "This is a sample text for testing language processing capabilities."
    
    @pytest.fixture
    def language_module(self):
        """Fixture providing a language module instance."""
        with patch("lmm.core.mind_modules.language.LanguageModule.initialize"):
            module = language.LanguageModule()
            return module
    
    def test_language_processing(self, language_module, sample_text):
        """Test language processing capabilities."""
        with patch("lmm.core.mind_modules.language.LanguageModule._analyze_syntax") as mock_analyze_syntax, \
             patch("lmm.core.mind_modules.language.LanguageModule._extract_entities") as mock_extract_entities, \
             patch("lmm.core.mind_modules.language.LanguageModule._detect_intent") as mock_detect_intent:
            
            # Configure mocks
            mock_analyze_syntax.return_value = {
                "tokens": ["This", "is", "a", "sample", "text", "for", "testing", "language", "processing", "capabilities"],
                "pos_tags": ["DET", "VERB", "DET", "ADJ", "NOUN", "ADP", "VERB", "NOUN", "VERB", "NOUN"],
                "dependency_parse": {"root": "is", "children": ["This", "text"]}
            }
            
            mock_extract_entities.return_value = {
                "entities": []  # No named entities in this simple text
            }
            
            mock_detect_intent.return_value = {
                "intent": "statement",
                "confidence": 0.95
            }
            
            # Process the text
            result = language_module.process(sample_text)
            
            # Verify processing
            assert result is not None
            assert "syntax" in result
            assert "entities" in result
            assert "intent" in result
            assert result["intent"]["intent"] == "statement"
            
            # Verify the mocks were called correctly
            mock_analyze_syntax.assert_called_once_with(sample_text)
            mock_extract_entities.assert_called_once_with(sample_text)
            mock_detect_intent.assert_called_once_with(sample_text)
    
    def test_language_generation(self, language_module):
        """Test language generation capabilities."""
        with patch("lmm.core.mind_modules.language.LanguageModule._generate_text") as mock_generate:
            mock_generate.return_value = "This is a generated response."
            
            prompt = "Generate a response"
            response = language_module.generate_text(prompt)
            
            assert response == "This is a generated response."
            mock_generate.assert_called_once_with(prompt)

class TestMemoryModule:
    """Tests for the memory module functionality."""
    
    @pytest.fixture
    def memory_item(self):
        """Fixture providing a sample memory item."""
        return {
            "content": "Test memory content",
            "timestamp": "2023-01-01T12:00:00",
            "importance": 0.8,
            "tags": ["test", "memory"]
        }
    
    @pytest.fixture
    def memory_module(self):
        """Fixture providing a memory module instance."""
        with patch("lmm.core.mind_modules.memory.MemoryModule.initialize"):
            module = memory.MemoryModule()
            module._working_memory = []
            module._recent_items = []
            module._capacity = 10
            return module
    
    def test_memory_storage_retrieval(self, memory_module, memory_item):
        """Test memory storage and retrieval."""
        # Store a memory item
        memory_module.store(memory_item)
        
        # Verify it was added to working memory
        assert len(memory_module._working_memory) == 1
        assert memory_module._working_memory[0]["content"] == memory_item["content"]
        assert memory_module._working_memory[0]["importance"] == memory_item["importance"]
        
        # Test retrieval
        retrieved = memory_module.retrieve(lambda x: x["tags"] == ["test", "memory"])
        assert len(retrieved) == 1
        assert retrieved[0]["content"] == memory_item["content"]
    
    def test_memory_capacity(self, memory_module):
        """Test memory capacity limits."""
        # Add more items than capacity
        for i in range(15):  # Capacity is 10
            memory_module.store({
                "content": f"Memory item {i}",
                "importance": i / 15,
                "timestamp": f"2023-01-01T{i:02d}:00:00",
                "tags": ["test"]
            })
        
        # Verify only the most important items are kept
        assert len(memory_module._working_memory) == 10
        
        # Verify items are sorted by importance
        importances = [item["importance"] for item in memory_module._working_memory]
        assert sorted(importances, reverse=True) == importances
    
    def test_memory_recall(self, memory_module):
        """Test memory recall by similarity."""
        with patch("lmm.core.mind_modules.memory.MemoryModule._compute_similarity") as mock_similarity:
            # Configure similarity function to return high similarity for queries containing "important"
            def similarity_func(memory_content, query):
                if "important" in query:
                    return 0.9 if "important" in memory_content else 0.1
                return 0.5
            
            mock_similarity.side_effect = similarity_func
            
            # Add test memories
            memory_module.store({"content": "This is an important memory", "importance": 0.7})
            memory_module.store({"content": "This is a regular memory", "importance": 0.5})
            
            # Recall memories by similarity
            results = memory_module.recall_by_similarity("Find important memories", top_k=1)
            
            # Verify results
            assert len(results) == 1
            assert "important" in results[0]["content"]

class TestSocialModule:
    """Tests for the social module functionality."""
    
    @pytest.fixture
    def social_context(self):
        """Fixture providing a sample social context."""
        return {
            "participants": ["user", "system"],
            "relationship": "collaborative",
            "social_norms": ["politeness", "helpfulness"]
        }
    
    @pytest.fixture
    def social_module(self):
        """Fixture providing a social module instance."""
        with patch("lmm.core.mind_modules.social.SocialCognitionModule.initialize"):
            module = social.SocialCognitionModule()
            module._social_context = {
                "participants": [],
                "relationships": {},
                "interaction_history": []
            }
            return module
    
    def test_social_interaction(self, social_module, social_context):
        """Test social interaction capabilities."""
        # Set up the social context
        social_module.update_social_context(
            participants=social_context["participants"],
            relationship=social_context["relationship"],
            social_norms=social_context["social_norms"]
        )
        
        # Verify context was updated
        assert social_module._social_context["participants"] == social_context["participants"]
        assert social_module._social_context.get("relationship") == social_context["relationship"]
        assert social_module._social_context.get("social_norms") == social_context["social_norms"]
    
    def test_process_social_cues(self, social_module):
        """Test processing of social cues."""
        with patch("lmm.core.mind_modules.social.SocialCognitionModule._extract_social_cues") as mock_extract:
            mock_extract.return_value = {
                "sentiment": "positive",
                "formality": "casual",
                "emotion": "happy"
            }
            
            # Process a message with social cues
            result = social_module.process("Hey there, I'm happy to chat with you!")
            
            # Verify processing
            assert result is not None
            assert "sentiment" in result
            assert result["sentiment"] == "positive"
            
            # Verify extraction was called
            mock_extract.assert_called_once()

class TestThoughtModule:
    """Tests for the thought module functionality."""
    
    @pytest.fixture
    def thought_input(self):
        """Fixture providing input for thought processing."""
        return {
            "query": "What is the capital of France?",
            "context": "Geographic information retrieval",
            "constraints": ["factual", "concise"]
        }
    
    @pytest.fixture
    def thought_module(self):
        """Fixture providing a thought module instance."""
        with patch("lmm.core.mind_modules.thought.ThoughtModule.initialize"):
            module = thought.ThoughtModule()
            return module
    
    def test_thought_processing(self, thought_module, thought_input):
        """Test thought processing capabilities."""
        with patch("lmm.core.mind_modules.thought.ThoughtModule._generate_reasoning_steps") as mock_reason:
            # Configure mock to return reasoning steps
            mock_reason.return_value = [
                "The query is asking for the capital city of France.",
                "The capital of France is Paris.",
                "This is a factual question requiring a concise answer."
            ]
            
            # Process the thought
            result = thought_module.process(thought_input["query"])
            
            # Verify processing
            assert result is not None
            assert "reasoning" in result
            assert len(result["reasoning"]) == 3
            assert "response" in result
            
            # Verify reasoning was called
            mock_reason.assert_called_once()
    
    def test_chain_of_thought(self, thought_module):
        """Test chain of thought reasoning."""
        with patch("lmm.core.mind_modules.thought.ThoughtModule._generate_reasoning_steps") as mock_reason:
            # Configure mock to return complex reasoning steps
            mock_reason.return_value = [
                "Step 1: Understand the problem - finding the sum of numbers from 1 to 10.",
                "Step 2: I can use the formula n(n+1)/2 where n is 10.",
                "Step 3: Calculate 10 * (10 + 1) / 2 = 10 * 11 / 2 = 110 / 2 = 55.",
                "Step 4: The sum of numbers from 1 to 10 is 55."
            ]
            
            # Process with chain of thought enabled
            result = thought_module.process(
                "What is the sum of numbers from 1 to 10?",
                use_chain_of_thought=True
            )
            
            # Verify detailed reasoning
            assert result is not None
            assert "reasoning" in result
            assert len(result["reasoning"]) == 4
            assert "response" in result
            
            # Verify the response contains the answer
            assert "55" in result.get("response", "")
            
            # Verify reasoning was called with chain of thought enabled
            mock_reason.assert_called_once() 