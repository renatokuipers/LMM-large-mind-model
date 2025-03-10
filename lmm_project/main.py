"""
Main entry point for the LMM system.
Initializes and starts the cognitive architecture.
"""
import os
import sys
import time
import logging
import argparse
from dotenv import load_dotenv
import yaml
import signal
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lmm_main")

# Import core components
from lmm_project.core import get_mind, ModuleType, StateError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Large Mind Model (LMM) System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--storage-dir", 
        type=str, 
        default=None, 
        help="Storage directory (overrides config and .env)"
    )
    parser.add_argument(
        "--load-state", 
        type=str, 
        default=None, 
        help="Load system state from file"
    )
    parser.add_argument(
        "--dev-rate", 
        type=float, 
        default=None, 
        help="Development rate (age units per second)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default=None, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Log level set to {log_level}")


def setup_environment():
    """Load environment variables and configuration."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = args.log_level or os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    # Determine configuration file path
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        config_path = None
    
    # Determine storage directory
    storage_dir = args.storage_dir or os.getenv("STORAGE_DIR", "storage")
    
    return args, config_path, storage_dir


def register_signal_handlers(mind):
    """Register signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, stopping the system...")
        mind.stop()
        logger.info("System stopped, exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the LMM system."""
    try:
        # Set up environment and configuration
        args, config_path, storage_dir = setup_environment()
        
        logger.info("Initializing LMM system...")
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Storage directory: {storage_dir}")
        
        # Initialize the mind
        mind = get_mind(config_path=config_path, storage_dir=storage_dir)
        
        # Register signal handlers
        register_signal_handlers(mind)
        
        # Override development rate if specified
        if args.dev_rate is not None:
            mind.development_rate = args.dev_rate
            logger.info(f"Development rate set to {args.dev_rate}")
        
        # Load state if specified
        if args.load_state:
            try:
                logger.info(f"Loading state from {args.load_state}")
                mind.load_state(args.load_state)
                logger.info("State loaded successfully")
            except StateError as e:
                logger.error(f"Failed to load state: {str(e)}")
                return 1
        
        # Start the system
        logger.info("Starting the LMM system...")
        mind.start()
        
        # Keep the main thread alive
        logger.info("LMM system running. Press Ctrl+C to stop.")
        
        # Main loop
        try:
            while True:
                time.sleep(1)
                
                # Periodic status logging (every 5 minutes)
                if int(time.time()) % 300 == 0:
                    age = mind.get_age()
                    stage = mind.get_developmental_stage()
                    logger.info(f"System running: Age = {age:.2f}, Stage = {stage.name}")
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping the system...")
        
        # Stop the system
        mind.stop()
        logger.info("LMM system stopped successfully")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
