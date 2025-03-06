"""
API endpoints for the Large Mind Model (LMM).

This module implements API endpoints for interacting with the LMM,
including conversation, status retrieval, and configuration.
"""
import os
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger

logger = get_logger("lmm.api.endpoints")

class LMMAPIHandler:
    """
    Handles API requests for the LMM.
    
    This class provides methods for handling API requests to interact
    with the LMM, including conversation, status retrieval, and configuration.
    """
    
    def __init__(self, lmm_instance=None):
        """
        Initialize the API Handler.
        
        Args:
            lmm_instance: Instance of the LargeMindsModel
        """
        self.lmm = lmm_instance
        logger.info("Initialized LMM API Handler")
    
    def handle_conversation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a conversation request.
        
        Args:
            request_data: Request data containing the message
            
        Returns:
            Response data
        """
        if not self.lmm:
            return {"error": "LMM instance not connected"}
        
        try:
            # Extract message from request
            message = request_data.get("message", "")
            if not message:
                return {"error": "No message provided"}
            
            # Process interaction
            response = self.lmm.interact(message)
            
            # Return response
            return {
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling conversation: {e}")
            return {"error": str(e)}
    
    def handle_status_request(self) -> Dict[str, Any]:
        """
        Handle a status request.
        
        Returns:
            Status data
        """
        if not self.lmm:
            return {"error": "LMM instance not connected"}
        
        try:
            # Get development status
            dev_status = self.lmm.get_development_status()
            
            # Get memory status
            memory_status = self.lmm.get_memory_status()
            
            # Return status
            return {
                "developmental_status": dev_status,
                "memory_status": memory_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            return {"error": str(e)}
    
    def handle_memory_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a memory request.
        
        Args:
            request_data: Request data containing the query
            
        Returns:
            Memory data
        """
        if not self.lmm:
            return {"error": "LMM instance not connected"}
        
        try:
            # Extract query from request
            query = request_data.get("query", "")
            if not query:
                return {"error": "No query provided"}
            
            # Get limit from request (default to 5)
            limit = request_data.get("limit", 5)
            
            # Recall memories
            memories = self.lmm.recall_memories(query, limit=limit)
            
            # Return memories
            return {
                "memories": memories,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling memory request: {e}")
            return {"error": str(e)}
    
    def handle_config_request(self, request_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a configuration request.
        
        This comprehensive implementation provides:
        1. Validation of configuration updates against Pydantic models
        2. Section-specific configuration updates
        3. Nested configuration property updates
        4. Schema information for configuration help
        5. Proper error reporting with validation details
        6. Configuration persistence options
        
        Args:
            request_data: Optional request data containing configuration updates
                - 'update': Dict of configuration updates
                - 'operation': Type of operation ('update', 'reset', 'save', 'load')
                - 'filepath': Optional filepath for save/load operations
                - 'section': Optional specific section to update or return
                - 'schema': Boolean to request schema information
            
        Returns:
            Dictionary with configuration data and operation results
        """
        try:
            # Get current configuration
            config = get_config()
            
            # Handle request with no data - return current configuration
            if not request_data:
                return self._get_config_response(config)
                
            # Process based on operation type
            operation = request_data.get('operation', 'update')
            
            # Configuration schema request
            if request_data.get('schema', False):
                return self._get_config_schema()
                
            # Handle different operations
            if operation == 'reset':
                return self._handle_config_reset(request_data.get('section'))
            elif operation == 'save':
                return self._handle_config_save(request_data.get('filepath', 'config.json'))
            elif operation == 'load':
                return self._handle_config_load(request_data.get('filepath', 'config.json'))
            elif operation == 'update':
                return self._handle_config_update(config, request_data)
            else:
                return {
                    "error": f"Unknown operation: {operation}",
                    "valid_operations": ["update", "reset", "save", "load"]
                }
                
        except Exception as e:
            logger.error(f"Error handling config request: {str(e)}")
            return {"error": str(e), "status": "failed"}
            
    def _get_config_response(self, config, section=None) -> Dict[str, Any]:
        """Generate a standard config response with optional section filtering."""
        from lmm.utils.config import LMMConfig
        
        # Convert config to dictionary
        config_dict = config.model_dump()
        
        # Filter by section if specified
        if section and section in config_dict:
            result = {section: config_dict[section]}
        else:
            result = config_dict
            
        return {
            "config": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """Return schema information for the configuration."""
        from lmm.utils.config import LMMConfig
        
        # Get schema from Pydantic model
        schema = LMMConfig.model_json_schema()
        
        # Add helpful metadata about each section
        section_help = {
            "llm": "LLM settings control language model connections and parameters",
            "memory": "Memory settings control vector storage and retrieval mechanisms",
            "mother": "Mother settings define the caregiver personality characteristics",
            "development": "Development settings control the growth and stage progression",
            "visualization": "Visualization settings for dashboard and metrics display"
        }
        
        return {
            "schema": schema,
            "section_help": section_help,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_config_reset(self, section=None) -> Dict[str, Any]:
        """Reset configuration to defaults."""
        from lmm.utils.config import LMMConfig, config
        
        # Create a new default config
        default_config = LMMConfig()
        
        # Reset either a specific section or the entire config
        if section and hasattr(config, section) and hasattr(default_config, section):
            # Reset just one section
            setattr(config, section, getattr(default_config, section))
            logger.info(f"Reset configuration section: {section}")
            return {
                "status": "success", 
                "message": f"Configuration section {section} reset to defaults",
                "config": {section: getattr(config, section).model_dump()}
            }
        elif section:
            return {"error": f"Unknown configuration section: {section}"}
        else:
            # Reset entire config
            from lmm.utils.config import config as global_config
            global_config = default_config
            logger.info("Reset entire configuration to defaults")
            return {
                "status": "success", 
                "message": "Entire configuration reset to defaults",
                "config": default_config.model_dump()
            }
    
    def _handle_config_save(self, filepath) -> Dict[str, Any]:
        """Save configuration to file."""
        from lmm.utils.config import save_config_to_file
        
        success = save_config_to_file(filepath)
        
        if success:
            return {
                "status": "success",
                "message": f"Configuration saved to {filepath}"
            }
        else:
            return {
                "status": "error",
                "error": f"Failed to save configuration to {filepath}"
            }
    
    def _handle_config_load(self, filepath) -> Dict[str, Any]:
        """Load configuration from file."""
        import os
        from lmm.utils.config import load_config_from_file, get_config
        
        if not os.path.exists(filepath):
            return {
                "status": "error",
                "error": f"Configuration file does not exist: {filepath}"
            }
        
        success = load_config_from_file(filepath)
        
        if success:
            return {
                "status": "success",
                "message": f"Configuration loaded from {filepath}",
                "config": get_config().model_dump()
            }
        else:
            return {
                "status": "error",
                "error": f"Failed to load configuration from {filepath}"
            }
    
    def _handle_config_update(self, current_config, request_data) -> Dict[str, Any]:
        """Update configuration with new values."""
        from lmm.utils.config import LMMConfig
        from pydantic import ValidationError
        
        # Extract update data and section filter
        updates = request_data.get('update', {})
        section = request_data.get('section')
        
        if not updates:
            return {
                "status": "error",
                "error": "No update data provided"
            }
        
        try:
            # Process updates for specific section
            if section:
                if not hasattr(current_config, section):
                    return {
                        "status": "error",
                        "error": f"Unknown configuration section: {section}"
                    }
                
                section_model = getattr(current_config.__class__, section).default_factory()
                section_data = getattr(current_config, section).model_dump()
                section_data.update(updates)
                
                # Validate updated section
                updated_section = section_model.__class__.model_validate(section_data)
                
                # Apply validated updates to the global config
                setattr(current_config, section, updated_section)
                
                logger.info(f"Updated configuration section: {section}")
                return {
                    "status": "success",
                    "message": f"Configuration section {section} updated",
                    "updated": updates,
                    "config": {section: getattr(current_config, section).model_dump()}
                }
            
            # Process full configuration update
            config_data = current_config.model_dump()
            
            # Apply updates to each section
            validation_errors = {}
            applied_updates = {}
            
            for section_name, section_updates in updates.items():
                if not hasattr(current_config, section_name):
                    validation_errors[section_name] = f"Unknown section: {section_name}"
                    continue
                
                try:
                    # Get current section data
                    section_model = getattr(current_config.__class__, section_name).default_factory()
                    section_data = getattr(current_config, section_name).model_dump()
                    
                    # Apply updates
                    if isinstance(section_updates, dict):
                        section_data.update(section_updates)
                    else:
                        validation_errors[section_name] = "Section updates must be a dictionary"
                        continue
                    
                    # Validate updated section
                    updated_section = section_model.__class__.model_validate(section_data)
                    
                    # Apply validated updates to the global config
                    setattr(current_config, section_name, updated_section)
                    applied_updates[section_name] = section_updates
                    logger.info(f"Updated configuration section: {section_name}")
                    
                except ValidationError as e:
                    validation_errors[section_name] = str(e)
            
            # Return results
            if validation_errors and not applied_updates:
                return {
                    "status": "error",
                    "error": "Configuration validation failed",
                    "validation_errors": validation_errors
                }
            elif validation_errors:
                return {
                    "status": "partial",
                    "message": "Some configuration updates applied with errors",
                    "applied_updates": applied_updates,
                    "validation_errors": validation_errors,
                    "config": current_config.model_dump()
                }
            else:
                return {
                    "status": "success",
                    "message": "Configuration updated successfully",
                    "applied_updates": applied_updates,
                    "config": current_config.model_dump()
                }
                
        except ValidationError as e:
            return {
                "status": "error",
                "error": "Configuration validation failed",
                "validation_details": str(e)
            }
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return {
                "status": "error",
                "error": f"Configuration update failed: {str(e)}"
            }
    
    def handle_set_stage_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request to set the developmental stage.
        
        Args:
            request_data: Request data containing the stage
            
        Returns:
            Response data
        """
        if not self.lmm:
            return {"error": "LMM instance not connected"}
        
        try:
            # Extract stage from request
            stage = request_data.get("stage", "")
            if not stage:
                return {"error": "No stage provided"}
            
            # Set stage
            self.lmm.set_developmental_stage(stage)
            
            # Return response
            return {
                "success": True,
                "stage": stage,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling set stage request: {e}")
            return {"error": str(e)}
    
    def handle_save_request(self) -> Dict[str, Any]:
        """
        Handle a request to save the LMM state.
        
        Returns:
            Response data
        """
        if not self.lmm:
            return {"error": "LMM instance not connected"}
        
        try:
            # Save state
            self.lmm.save_state()
            
            # Return response
            return {
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling save request: {e}")
            return {"error": str(e)} 