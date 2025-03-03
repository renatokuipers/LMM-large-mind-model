# utils/file_manager.py
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FileManager:
    """Handles file system operations for the generator"""
    
    def __init__(self, base_dir: str):
        """Initialize with the base output directory"""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file, creating directories as needed"""
        try:
            # Convert to absolute path if it's relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.base_dir, file_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully wrote file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {str(e)}")
            return False
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file"""
        try:
            # Convert to absolute path if it's relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.base_dir, file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return ""
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""
    
    def copy_template(self, template_name: str, destination: str) -> bool:
        """Copy a project template to the destination"""
        try:
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'project_structures', template_name)
            
            # Convert destination to absolute path if it's relative
            if not os.path.isabs(destination):
                destination = os.path.join(self.base_dir, destination)
            
            # Check if template exists
            if not os.path.exists(template_dir):
                logger.error(f"Template does not exist: {template_name}")
                return False
            
            # Copy template directory
            shutil.copytree(template_dir, destination, dirs_exist_ok=True)
            
            logger.info(f"Successfully copied template {template_name} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying template {template_name}: {str(e)}")
            return False