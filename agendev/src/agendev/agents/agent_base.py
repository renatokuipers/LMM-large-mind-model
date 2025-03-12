# agent_base.py
"""Base agent class for all specialized agents in the AgenDev system."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from ..llm_integration import LLMIntegration, LLMConfig
from ..context_management import ContextManager


class AgentStatus(str, Enum):
    """Status states for agents."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SUCCESS = "success"


class AgentAction(BaseModel):
    """Represents an action performed by an agent."""
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    action_type: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class Agent(BaseModel):
    """Base class for all AgenDev agents."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    agent_type: str
    status: AgentStatus = AgentStatus.IDLE
    last_action: Optional[AgentAction] = None
    action_history: List[UUID] = Field(default_factory=list)
    
    # Shared resources
    llm: Optional[LLMIntegration] = None
    context_manager: Optional[ContextManager] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def initialize(self, llm: LLMIntegration, context_manager: Optional[ContextManager] = None) -> None:
        """
        Initialize the agent with shared resources.
        
        Args:
            llm: LLM integration for agent communication
            context_manager: Optional context manager for accessing project context
        """
        self.llm = llm
        self.context_manager = context_manager
        self.status = AgentStatus.IDLE
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return a result. Must be implemented by subclasses.
        
        Args:
            input_data: Input data for the agent to process
            
        Returns:
            Dictionary with the processing result
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_action(self, action_type: str, description: str, parameters: Dict[str, Any]) -> AgentAction:
        """
        Create a new action record.
        
        Args:
            action_type: Type of action
            description: Description of the action
            parameters: Parameters for the action
            
        Returns:
            Created action record
        """
        action = AgentAction(
            agent_id=self.id,
            action_type=action_type,
            description=description,
            parameters=parameters,
            status=AgentStatus.BUSY
        )
        self.action_history.append(action.id)
        self.last_action = action
        self.status = AgentStatus.BUSY
        return action
    
    def complete_action(self, action: AgentAction, result: Dict[str, Any], status: AgentStatus = AgentStatus.SUCCESS) -> AgentAction:
        """
        Mark an action as complete.
        
        Args:
            action: Action to complete
            result: Result of the action
            status: Final status of the action
            
        Returns:
            Updated action record
        """
        action.result = result
        action.status = status
        action.completed_at = datetime.now()
        self.status = AgentStatus.IDLE if status == AgentStatus.SUCCESS else status
        return action
    
    def get_action_history(self) -> List[AgentAction]:
        """
        Get the action history for this agent.
        
        Returns:
            List of actions performed by this agent
        """
        return [self.last_action] if self.last_action else [] 