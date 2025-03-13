# Simple LLM API test script
import requests
import json
import sys
import traceback
from typing import List, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_llm_api(
    base_url: str = "http://192.168.2.12:1234",
    model: str = "qwen2.5-7b-instruct"
) -> bool:
    """Simple test to check if the LLM API is working correctly."""
    logger.info(f"Testing LLM API connection to {base_url}")
    
    # Test API health
    try:
        logger.info("Testing API health...")
        health_url = f"{base_url}/v1/health"
        response = requests.get(health_url)
        logger.info(f"Health check response: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")
        return False
    
    # Test chat completion
    try:
        logger.info("Testing chat completion...")
        chat_url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # Minimal payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Say hello world"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        logger.info(f"Sending request with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(chat_url, headers=headers, json=payload)
        
        logger.info(f"Response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Chat completion request failed: {response.text}")
            return False
        
        # Parse response
        try:
            response_json = response.json()
            logger.info(f"Response keys: {list(response_json.keys())}")
            
            if "choices" not in response_json:
                logger.error("Missing 'choices' in response")
                return False
            
            content = response_json["choices"][0]["message"]["content"]
            logger.info(f"Received response: {content}")
            
            return True
            
        except Exception as parse_error:
            logger.error(f"Error parsing response: {parse_error}")
            logger.error(f"Raw response: {response.text}")
            return False
        
    except Exception as e:
        logger.error(f"Error during chat completion test: {e}")
        traceback.print_exc()
        return False

def test_with_different_params():
    """Test various parameter combinations to find a working setup."""
    model_options = ["qwen2.5-7b-instruct", "llama2-7b", "gpt-3.5-turbo"]
    temperature_options = [0.2, 0.7, 1.0]
    
    for model in model_options:
        for temp in temperature_options:
            logger.info(f"Testing with model={model}, temperature={temp}")
            
            try:
                url = "http://192.168.2.12:1234"
                headers = {"Content-Type": "application/json"}
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": "Say hello world"}
                    ],
                    "temperature": temp
                }
                
                response = requests.post(f"{url}/v1/chat/completions", 
                                        headers=headers, 
                                        json=payload,
                                        timeout=10)
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    logger.info(f"Success with model={model}, temperature={temp}")
                    logger.info(f"Response: {response.json()}")
                    return True
                else:
                    logger.warning(f"Failed with {model}, {temp}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error with {model}, {temp}: {e}")
                continue
    
    return False

if __name__ == "__main__":
    logger.info("Starting LLM API test")
    
    success = test_llm_api()
    if success:
        logger.info("🎉 LLM API test PASSED! The API is working correctly.")
    else:
        logger.error("❌ LLM API test FAILED! There may be issues with the API connection.")
        logger.info("Trying alternative parameters...")
        
        if test_with_different_params():
            logger.info("Found working parameters! Please update your LLM config accordingly.")
        else:
            logger.error("All parameter combinations failed. Please check your API endpoint.") 