import os
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
import logging

from lmm_project.core.mind import Mind
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.utils.llm_client import LLMClient
from lmm_project.utils.tts_client import TTSClient
from lmm_project.core.exceptions import LMMError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lmm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LMM")

def load_config(config_path: str = "config.yml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"development_mode": True}

def initialize_clients(config: dict) -> tuple:
    """Initialize API clients"""
    # Get API URLs from environment or config
    llm_api_url = os.environ.get("LLM_API_URL", config.get("llm_api_url", "http://192.168.2.12:1234"))
    tts_api_url = os.environ.get("TTS_API_URL", config.get("tts_api_url", "http://127.0.0.1:7860"))
    
    # Initialize clients
    llm_client = LLMClient(base_url=llm_api_url)
    
    # TTS client is optional
    tts_client = None
    try:
        tts_client = TTSClient(base_url=tts_api_url)
    except Exception as e:
        logger.warning(f"TTS client initialization failed: {e}")
    
    return llm_client, tts_client

def create_storage_directories():
    """Create necessary storage directories"""
    directories = [
        "storage/states",
        "storage/conversations",
        "storage/memories",
        "storage/embeddings",
        "generated"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """Main entry point for the LMM application"""
    parser = argparse.ArgumentParser(description="Large Mind Model")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--load-state", type=str, help="Load mind state from file")
    parser.add_argument("--development-rate", type=float, default=0.01, help="Development rate per cycle")
    parser.add_argument("--cycles", type=int, default=100, help="Number of development cycles to run")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create storage directories
    create_storage_directories()
    
    # Initialize clients
    llm_client, tts_client = initialize_clients(config)
    
    # Initialize Mother LLM
    logger.info("Initializing Mother LLM...")
    mother = MotherLLM(
        llm_client=llm_client, 
        tts_client=tts_client,
        teaching_style=config.get("teaching_style", "socratic"),
        voice=config.get("mother_voice", "af_bella")
    )
    
    # Initialize Mind
    logger.info("Initializing Large Mind Model...")
    mind = Mind()
    
    # Load state if specified
    if args.load_state:
        try:
            logger.info(f"Loading mind state from {args.load_state}")
            mind.state_manager.load_state(args.load_state)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    # Initialize modules
    mind.initialize_modules()
    logger.info(f"Mind initialized with {len(mind.modules)} modules")
    
    # Development loop
    logger.info(f"Starting development with {args.cycles} cycles at rate {args.development_rate}")
    try:
        for cycle in range(args.cycles):
            # Update development
            development_result = mind.update_development(args.development_rate)
            
            # Log progress every 10 cycles
            if cycle % 10 == 0:
                logger.info(f"Cycle {cycle}: Age {development_result['age']:.2f}, Stage: {development_result['stage']}")
                for module_name, level in development_result['modules'].items():
                    logger.info(f"  - {module_name}: {level:.2f}")
            
            # Save state every 50 cycles
            if cycle % 50 == 0 and cycle > 0:
                state_path = mind.state_manager.save_state()
                logger.info(f"Saved state to {state_path}")
            
            # Small delay to prevent resource hogging
            time.sleep(0.1)
        
        # Final state save
        state_path = mind.state_manager.save_state()
        logger.info(f"Development complete. Final state saved to {state_path}")
        
    except KeyboardInterrupt:
        logger.info("Development interrupted by user")
        state_path = mind.state_manager.save_state()
        logger.info(f"State saved to {state_path}")
    except LMMError as e:
        logger.error(f"LMM error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    logger.info("LMM session ended")

if __name__ == "__main__":
    main() 
