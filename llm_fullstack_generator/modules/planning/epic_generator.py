# modules/planning/epic_generator.py
from typing import List
from core.schemas import Epic, ProjectConfig
from utils.llm_client import LLMClient, Message
import logging

logger = logging.getLogger(__name__)

class EpicGenerator:
    """Generates structured EPIC tasks for project planning"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_epics(self, project_name: str, project_description: str, config: ProjectConfig) -> List[Epic]:
        """Generate EPICs and tasks for the project"""
        logger.info(f"Generating EPICs for project: {project_name}")
        
        try:
            epics = self.llm_client.generate_epics(project_name, project_description, config)
            logger.info(f"Generated {len(epics)} EPICs with {sum(len(epic.tasks) for epic in epics)} tasks")
            return epics
        except Exception as e:
            logger.error(f"Error generating EPICs: {str(e)}")
            raise