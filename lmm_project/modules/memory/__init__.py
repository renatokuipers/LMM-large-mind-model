# Memory module 

from typing import Optional, Dict, Any

from lmm_project.core.event_bus import EventBus
from lmm_project.modules.memory.working_memory import WorkingMemory
from lmm_project.modules.memory.long_term_memory import LongTermMemory
from lmm_project.modules.memory.semantic_memory import SemanticMemoryModule
from lmm_project.modules.memory.episodic_memory import EpisodicMemoryModule
from lmm_project.modules.memory.associative_memory import AssociativeMemoryModule

def get_module(module_id: str, event_bus: Optional[EventBus] = None) -> Any:
    """
    Factory function to create a memory module.
    
    The memory system integrates multiple specialized memory modules:
    - Working memory (short-term, limited capacity buffer)
    - Long-term memory (persistent storage)
    - Semantic memory (concepts and knowledge)
    - Episodic memory (experiences and events)
    - Associative memory (links between memories)
    
    Returns:
    An instance of MemorySystem
    """
    # Create the memory system with all sub-modules
    return MemorySystem(
        module_id=module_id,
        event_bus=event_bus
    )

class MemorySystem:
    """
    Integrated memory system combining all memory types
    
    This class serves as a facade over the different memory sub-systems,
    providing a unified interface while delegating operations to the
    appropriate specialized module.
    """
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        # Initialize all memory modules
        self.module_id = module_id
        self.module_type = "memory"
        self.event_bus = event_bus
        
        # Create sub-modules
        self.working_memory = WorkingMemory(f"{module_id}_working", event_bus)
        self.long_term_memory = LongTermMemory(f"{module_id}_longterm", event_bus)
        self.semantic_memory = SemanticMemoryModule(f"{module_id}_semantic", event_bus)
        self.episodic_memory = EpisodicMemoryModule(f"{module_id}_episodic", event_bus)
        self.associative_memory = AssociativeMemoryModule(f"{module_id}_associative", event_bus)
        
        # Track development level as average of sub-modules
        self.development_level = 0.0
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the appropriate memory subsystem
        
        Parameters:
        input_data: Dictionary with:
            - system: Which memory system to use (working, longterm, semantic, episodic, associative)
            - operation: The operation to perform
            - Additional parameters specific to the operation
            
        Returns:
        The result from the appropriate subsystem
        """
        system = input_data.get("system", "working")
        
        # Route to appropriate subsystem
        if system == "working":
            return self.working_memory.process_input(input_data)
        elif system == "longterm":
            return self.long_term_memory.process_input(input_data)
        elif system == "semantic":
            return self.semantic_memory.process_input(input_data)
        elif system == "episodic":
            return self.episodic_memory.process_input(input_data)
        elif system == "associative":
            return self.associative_memory.process_input(input_data)
        else:
            return {"status": "error", "message": f"Unknown memory system: {system}"}
    
    def update_development(self, amount: float) -> float:
        """
        Update development of all memory subsystems
        
        Parameters:
        amount: Development amount to apply
        
        Returns:
        Average development level across subsystems
        """
        # Update each subsystem
        working_level = self.working_memory.update_development(amount)
        longterm_level = self.long_term_memory.update_development(amount)
        semantic_level = self.semantic_memory.update_development(amount)
        episodic_level = self.episodic_memory.update_development(amount)
        associative_level = self.associative_memory.update_development(amount)
        
        # Calculate average development level
        self.development_level = (
            working_level + 
            longterm_level + 
            semantic_level + 
            episodic_level + 
            associative_level
        ) / 5.0
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the memory system"""
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "working_memory": self.working_memory.get_state(),
            "long_term_memory": self.long_term_memory.get_state(),
            "semantic_memory": self.semantic_memory.get_state(),
            "episodic_memory": self.episodic_memory.get_state(),
            "associative_memory": self.associative_memory.get_state()
        }
    
    def save_state(self) -> Dict[str, str]:
        """Save the state of all memory subsystems"""
        return {
            "working_memory": self.working_memory.save_state(),
            "long_term_memory": self.long_term_memory.save_state(),
            "semantic_memory": self.semantic_memory.save_state(),
            "episodic_memory": self.episodic_memory.save_state(),
            "associative_memory": self.associative_memory.save_state()
        } 