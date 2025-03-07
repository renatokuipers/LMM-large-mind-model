import pytest
from unittest.mock import patch, MagicMock
from lmm.api import endpoints

class TestEndpoints:
    @pytest.fixture
    def sample_request_data(self):
        """Fixture providing sample request data for testing endpoints."""
        return {
            "message": "test query",
            "parameters": {"param1": "value1", "param2": "value2"}
        }
    
    @pytest.fixture
    def mock_lmm_instance(self):
        """Fixture providing a mock LLM instance."""
        mock_lmm = MagicMock()
        mock_lmm.interact.return_value = "This is a test response"
        mock_lmm.get_development_status.return_value = {"stage": "test_stage", "progress": 0.5}
        mock_lmm.get_memory_status.return_value = {"total_memories": 10, "types": {"episodic": 5, "semantic": 5}}
        mock_lmm.set_developmental_stage.return_value = None
        return mock_lmm
    
    def test_endpoint_initialization(self):
        """Test that endpoints can be properly initialized."""
        handler = endpoints.LMMAPIHandler()
        assert handler is not None
        assert hasattr(handler, "handle_conversation")
        assert hasattr(handler, "handle_status_request")
        assert hasattr(handler, "handle_memory_request")
        assert hasattr(handler, "handle_config_request")
    
    def test_endpoint_response(self, sample_request_data, mock_lmm_instance):
        """Test endpoint response with sample data."""
        handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
        response = handler.handle_conversation(sample_request_data)
        
        assert "response" in response
        assert response["response"] == "This is a test response"
        assert "timestamp" in response
        mock_lmm_instance.interact.assert_called_once_with(sample_request_data["message"], stream=False)
    
    def test_status_endpoint(self, mock_lmm_instance):
        """Test status endpoint."""
        handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
        response = handler.handle_status_request()
        
        assert "status" in response
        assert "development" in response
        assert response["development"]["stage"] == "test_stage"
        assert response["development"]["progress"] == 0.5
        assert "memory" in response
        assert response["memory"]["total_memories"] == 10
        mock_lmm_instance.get_development_status.assert_called_once()
        mock_lmm_instance.get_memory_status.assert_called_once()
    
    def test_memory_endpoint(self, mock_lmm_instance):
        """Test memory endpoint."""
        mock_lmm_instance.recall_memories.return_value = [
            {"id": 1, "content": "test memory 1", "type": "episodic"},
            {"id": 2, "content": "test memory 2", "type": "semantic"}
        ]
        
        handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
        response = handler.handle_memory_request({"query": "test query", "limit": 2})
        
        assert "memories" in response
        assert len(response["memories"]) == 2
        assert response["memories"][0]["id"] == 1
        assert response["memories"][1]["content"] == "test memory 2"
        mock_lmm_instance.recall_memories.assert_called_once_with(
            query="test query", limit=2, memory_type=None, 
            min_activation=0.0, context_tags=None, retrieval_strategy="combined"
        )
    
    @pytest.mark.parametrize("input_data,expected_status", [
        ({"message": "valid query"}, 200),
        ({"message": ""}, 400),
        (None, 400)
    ])
    def test_endpoint_validation(self, input_data, expected_status, mock_lmm_instance):
        """Test endpoint validation with various input scenarios."""
        handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
        
        if expected_status == 200:
            response = handler.handle_conversation(input_data)
            assert "response" in response
            assert "timestamp" in response
        else:
            response = handler.handle_conversation(input_data)
            assert "error" in response
            if input_data is None:
                assert "Invalid request data" in response["error"]
            elif "message" in input_data and input_data["message"] == "":
                assert "Empty message" in response["error"]
    
    def test_config_endpoint_get(self, mock_lmm_instance):
        """Test configuration endpoint get operation."""
        with patch("lmm.api.endpoints.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = {
                "llm": {"base_url": "test_url", "temperature": 0.7},
                "memory": {"vector_db_path": "test_path"}
            }
            mock_get_config.return_value = mock_config
            
            handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
            response = handler.handle_config_request()
            
            assert "config" in response
            assert response["config"]["llm"]["base_url"] == "test_url"
            assert response["config"]["memory"]["vector_db_path"] == "test_path"
    
    def test_set_stage_endpoint(self, mock_lmm_instance):
        """Test set stage endpoint."""
        handler = endpoints.LMMAPIHandler(lmm_instance=mock_lmm_instance)
        response = handler.handle_set_stage_request({"stage": "test_stage"})
        
        assert "status" in response
        assert response["status"] == "success"
        assert "message" in response
        mock_lmm_instance.set_developmental_stage.assert_called_once_with("test_stage") 