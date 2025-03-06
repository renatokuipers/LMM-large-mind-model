"""
Main application entry point for the Large Mind Model (LMM).

This module ties together all the components of the LMM system and
provides the main application logic.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from lmm.utils.config import get_config, load_config_from_dict
from lmm.utils.logging import get_logger, setup_logger
from lmm.core.mother.caregiver import MotherCaregiver
from lmm.core.development.stages import DevelopmentalStageManager
from lmm.core.development.learning import LearningManager
from lmm.memory.persistence import MemoryManager, MemoryType, MemoryImportance
from lmm.memory.advanced_memory import AdvancedMemoryManager

# Import mind modules
from lmm.core.mind_modules.emotion import EmotionModule
from lmm.core.mind_modules.language import LanguageModule
from lmm.core.mind_modules.memory import MemoryModule
from lmm.core.mind_modules.social import SocialCognitionModule
from lmm.core.mind_modules.consciousness import ConsciousnessModule

logger = get_logger("lmm.main")

class LargeMindsModel:
    """
    Large Mind Model (LMM) main class.
    
    This class coordinates all the components of the LMM system and
    provides the main application API.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the LMM.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        # Load configuration
        if config_dict:
            load_config_from_dict(config_dict)
        config = get_config()
        
        # Initialize components
        self.mother = MotherCaregiver()
        self.stage_manager = DevelopmentalStageManager()
        self.learning_manager = LearningManager()
        
        # Initialize mind modules
        self.memory_module = MemoryModule()
        self.emotional_module = EmotionModule()
        self.language_module = LanguageModule()
        self.social_module = SocialCognitionModule()
        self.consciousness_module = ConsciousnessModule()
        
        # Track interactions
        self.interaction_count = 0
        
        logger.info("Initialized Large Mind Model")
    
    def interact(self, message: str, stream: bool = False) -> str:
        """
        Process an interaction with the LMM.
        
        Args:
            message: Message to process
            stream: Whether to stream the response
            
        Returns:
            Response from the LMM
        """
        # Increment interaction count
        self.interaction_count += 1
        current_stage = self.stage_manager.get_current_stage()
        logger.info(f"Processing interaction {self.interaction_count} in stage {current_stage}")
        
        # Get emotional state before processing
        emotional_state = self.emotional_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Store user message in memory
        message_memory = {
            "operation": "store",
            "parameters": {
                "content": f"User: {message}",
                "memory_type": MemoryType.EPISODIC.value,
                "importance": MemoryImportance.MEDIUM.value,
                "context_tags": ["user_message"],
                "metadata": {
                    "interaction_number": self.interaction_count,
                    "emotional_state": emotional_state.get("state", {})
                }
            },
            "developmental_stage": current_stage
        }
        self.memory_module.process(message_memory)
        
        # Retrieve relevant memories for context
        memory_search = {
            "operation": "search",
            "parameters": {
                "query": message,
                "limit": 5,
                "min_activation": 0.3,
                "retrieval_strategy": "combined"
            },
            "developmental_stage": current_stage
        }
        memory_result = self.memory_module.process(memory_search)
        relevant_memories = memory_result.get("memories", [])
        
        # Process language understanding
        language_result = self.language_module.process({
            "input": message,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Process social cognition
        social_result = self.social_module.process({
            "input": message,
            "language_understanding": language_result,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Process consciousness (self-awareness, reflection)
        consciousness_result = self.consciousness_module.process({
            "input": message,
            "language_understanding": language_result,
            "social_understanding": social_result,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Generate response from mother based on developmental stage
        try:
            response = self.mother.respond(
                message=message,
                stage=current_stage,
                language_understanding=language_result,
                social_understanding=social_result,
                consciousness_state=consciousness_result,
                memories=relevant_memories,
                emotional_state=emotional_state.get("state", {}),
                stream=stream
            )
            
            # Update emotional state after processing
            self.emotional_module.process({
                "operation": "update",
                "input": message,
                "response": response,
                "developmental_stage": current_stage
            })
            
            # Store response in memory
            response_memory = {
                "operation": "store",
                "parameters": {
                    "content": f"LMM: {response}",
                    "memory_type": MemoryType.EPISODIC.value,
                    "importance": MemoryImportance.MEDIUM.value,
                    "context_tags": ["lmm_response"],
                    "related_memories": [m.get("id") for m in relevant_memories if m.get("id")],
                    "metadata": {
                        "interaction_number": self.interaction_count,
                        "language_understanding": language_result,
                        "social_understanding": social_result,
                        "consciousness_state": consciousness_result
                    }
                },
                "developmental_stage": current_stage
            }
            self.memory_module.process(response_memory)
            
            # Store interaction in memory for potential semantic memory formation
            self._store_interaction_memory(message, response, current_stage)
            
            # Update learning metrics based on interaction
            self.learning_manager.update_metrics(
                interaction_count=self.interaction_count,
                message=message,
                response=response,
                language_understanding=language_result,
                social_understanding=social_result,
                consciousness_state=consciousness_result,
                emotional_state=emotional_state.get("state", {}),
                developmental_stage=current_stage
            )
            
            # Check for stage progression
            self.stage_manager.check_progression(self.learning_manager.get_metrics())
            
            return response
        except Exception as e:
            logger.error(f"Error in interaction: {str(e)}")
            return f"Error: {str(e)}"
    
    def _store_interaction_memory(self, message: str, response: str, current_stage: str) -> None:
        """
        Store an interaction in memory.
        
        Args:
            message: User message
            response: LMM response
            current_stage: Current developmental stage
        """
        # Store conversation as episodic memory
        episodic_memory = {
            "operation": "store",
            "parameters": {
                "content": f"Conversation:\nUser: {message}\nLMM: {response}",
                "memory_type": MemoryType.EPISODIC.value,
                "importance": MemoryImportance.MEDIUM.value,
                "context_tags": ["conversation", f"interaction_{self.interaction_count}"],
                "metadata": {
                    "interaction_number": self.interaction_count,
                    "timestamp": datetime.now().isoformat()
                }
            },
            "developmental_stage": current_stage
        }
        self.memory_module.process(episodic_memory)
        
        # Extract potential semantic memories
        # Here we'd use more sophisticated NLP to identify key concepts
        # For now, use a simple approach for demonstration
        if len(message) > 30:
            semantic_memory = {
                "operation": "store",
                "parameters": {
                    "content": f"Learned from conversation: {message[:100]}...",
                    "memory_type": MemoryType.SEMANTIC.value,
                    "importance": MemoryImportance.LOW.value,
                    "context_tags": ["learning", "conversation_derived"],
                    "metadata": {
                        "source_interaction": self.interaction_count
                    }
                },
                "developmental_stage": current_stage
            }
            self.memory_module.process(semantic_memory)
    
    def get_development_status(self) -> Dict[str, Any]:
        """
        Get the current developmental status of the LMM.
        
        Returns:
            Dictionary with developmental status
        """
        # Get stage info
        stage_info = self.stage_manager.get_status()
        
        # Get learning analytics
        learning_metrics = self.learning_manager.get_metrics()
        
        # Combine information
        status = {
            "current_stage": stage_info["current_stage"],
            "stage_progress": stage_info["stage_progress"],
            "overall_progress": stage_info["overall_progress"],
            "interaction_count": self.interaction_count,
            "learning_metrics": learning_metrics,
            "brain_development": {
                "language_capacity": learning_metrics.get("language_complexity", 0),
                "emotional_awareness": learning_metrics.get("emotional_awareness", 0),
                "social_understanding": learning_metrics.get("social_understanding", 0),
                "cognitive_capability": learning_metrics.get("cognitive_capability", 0),
                "self_awareness": learning_metrics.get("self_awareness", 0)
            }
        }
        
        return status
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get the status of the memory system.
        
        Returns:
            Dictionary with memory status
        """
        memory_stats = self.memory_module.process({
            "operation": "get_stats"
        })
        
        working_memory = self.memory_module.process({
            "operation": "get_working_memory"
        })
        
        return {
            "memory_stats": memory_stats.get("stats", {}),
            "working_memory": working_memory.get("contents", [])
        }
    
    def get_mind_modules_status(self) -> Dict[str, Any]:
        """
        Get the status of all mind modules.
        
        Returns:
            Dictionary with mind modules status
        """
        modules_status = {
            "memory": self.memory_module.get_module_status(),
            "emotional": self.emotional_module.get_module_status(),
            "language": self.language_module.get_module_status(),
            "social": self.social_module.get_module_status(),
            "consciousness": self.consciousness_module.get_module_status()
        }
        
        return modules_status
    
    def recall_memories(
        self, 
        query: str, 
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_activation: float = 0.0,
        context_tags: Optional[List[str]] = None,
        retrieval_strategy: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Recall memories based on query.
        
        Args:
            query: Search query
            memory_type: Optional memory type filter
            limit: Maximum number of results
            min_activation: Minimum memory activation level
            context_tags: Optional context tags to filter by
            retrieval_strategy: Strategy for retrieval (vector, graph, context, combined)
            
        Returns:
            List of matching memories
        """
        # Get current stage
        current_stage = self.stage_manager.get_current_stage()
        
        # Search memories
        memory_search = {
            "operation": "search",
            "parameters": {
                "query": query,
                "memory_type": memory_type,
                "limit": limit,
                "min_activation": min_activation,
                "context_tags": context_tags,
                "retrieval_strategy": retrieval_strategy
            },
            "developmental_stage": current_stage
        }
        
        memory_result = self.memory_module.process(memory_search)
        memories = memory_result.get("memories", [])
        
        # Format output for external use
        formatted_memories = []
        for memory in memories:
            formatted_memory = {
                "id": memory.get("id"),
                "content": memory.get("content"),
                "type": memory.get("type"),
                "importance": memory.get("importance"),
                "created_at": memory.get("created_at"),
                "retrieval_score": memory.get("retrieval_score", 0.0)
            }
            
            # Add reconstruction information if available
            if memory.get("reconstructed"):
                formatted_memory["reconstructed"] = True
                formatted_memory["confidence"] = memory.get("confidence", 0.5)
                
            formatted_memories.append(formatted_memory)
        
        # Update retrieval statistics for visualization
        self.update_retrieval_stats(query, formatted_memories)
        
        return formatted_memories
    
    def get_introspection(self) -> str:
        """
        Get introspective information about the LMM's current state.
        
        Returns:
            Introspective response
        """
        # Get current stage and status
        current_stage = self.stage_manager.get_current_stage()
        development_status = self.get_development_status()
        memory_status = self.get_memory_status()
        
        # Get cognitive metrics from learning manager
        learning_metrics = self.learning_manager.get_metrics()
        cognitive_capacity = learning_metrics.get("cognitive_capacity", 0.0)
        attention_focus = learning_metrics.get("current_attention_focus", "unknown")
        cognitive_load = learning_metrics.get("cognitive_load", 0.0)
        
        # Get consciousness state
        consciousness_state = self.consciousness_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Get emotional state
        emotional_state = self.emotional_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Construct introspection
        introspection = f"Current developmental stage: {current_stage}\n\n"
        
        # Add cognitive metrics
        introspection += "Cognitive Status:\n"
        introspection += f"- Cognitive capacity: {cognitive_capacity:.2f}\n"
        introspection += f"- Current attention focus: {attention_focus}\n"
        introspection += f"- Cognitive load: {cognitive_load:.2f}\n"
        
        # Add consciousness insights
        introspection += "\nSelf-awareness Insights:\n"
        for insight in consciousness_state.get("recent_insights", [])[:3]:
            introspection += f"- {insight}\n"
        
        # Add emotional state
        emotions = emotional_state.get("state", {})
        introspection += "\nEmotional State:\n"
        for emotion, intensity in emotions.items():
            if intensity > 0.1:
                introspection += f"- {emotion}: {intensity:.2f}\n"
        
        # Add memory status
        working_memory = memory_status.get("working_memory", [])
        if working_memory:
            introspection += "\nCurrently in working memory:\n"
            for memory in working_memory[:3]:
                introspection += f"- {memory.get('content', '')[:50]}...\n"
        
        return introspection
    
    def simulate_development(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Simulate development over time to observe learning patterns.
        
        Args:
            iterations: Number of iterations to simulate
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Simulating development over {iterations} iterations")
        
        # Get current stage
        current_stage = self.stage_manager.get_current_stage()
        
        # Use learning manager to simulate development
        simulation_results = self.learning_manager.simulate_development(iterations)
        
        # After simulation, consolidate memories with findings
        consolidation_results = self.memory_module.process({
            "operation": "consolidate",
            "parameters": {"force": True},
            "developmental_stage": current_stage
        })
        
        # Add memory consolidation to results
        simulation_results["memory_consolidation"] = {
            "consolidated_count": consolidation_results.get("consolidated_count", 0)
        }
        
        return simulation_results
    
    def save_state(self) -> None:
        """Save the current state of the LMM."""
        # This would save all component states to persistent storage
        logger.info("Saved LMM state")
    
    def set_developmental_stage(self, stage: str) -> None:
        """
        Set the developmental stage manually.
        
        Args:
            stage: Stage to set
        """
        self.stage_manager.set_stage(stage)
        
        # Update memory parameters for new stage
        self.memory_module.process({
            "developmental_stage": stage
        })
        
        logger.info(f"Manually set developmental stage to {stage}")

    def get_memory_graph(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the memory association graph data.
        
        Args:
            limit: Maximum number of nodes to include
            
        Returns:
            Dictionary with memory graph data
        """
        # Get memory module to access advanced memory features
        if hasattr(self, "memory_module"):
            memory_graph_result = self.memory_module.process({
                "operation": "get_memory_graph",
                "parameters": {"limit": limit}
            })
            
            if memory_graph_result.get("success"):
                return memory_graph_result.get("graph", {})
        
        # Return empty graph if memory module is not available
        return {"nodes": [], "edges": []}

    def update_retrieval_stats(self, query: str, retrieved_memories: List[Dict[str, Any]]) -> None:
        """
        Update memory retrieval statistics.
        
        Args:
            query: The search query
            retrieved_memories: List of retrieved memories
        """
        # This method would be used by any visualization tool to track retrieval patterns
        avg_score = 0.0
        if retrieved_memories:
            avg_score = sum(memory.get("retrieval_score", 0.0) for memory in retrieved_memories) / len(retrieved_memories)
        
        # Store retrieval stats for visualization
        if hasattr(self, "_retrieval_stats"):
            self._retrieval_stats["counts"].append(len(retrieved_memories))
            self._retrieval_stats["scores"].append(avg_score)
            self._retrieval_stats["timestamps"].append(datetime.now())
            self._retrieval_stats["queries"].append(query)
            
            # Limit history size
            if len(self._retrieval_stats["counts"]) > 100:
                self._retrieval_stats["counts"] = self._retrieval_stats["counts"][-100:]
                self._retrieval_stats["scores"] = self._retrieval_stats["scores"][-100:]
                self._retrieval_stats["timestamps"] = self._retrieval_stats["timestamps"][-100:]
                self._retrieval_stats["queries"] = self._retrieval_stats["queries"][-100:]
        else:
            self._retrieval_stats = {
                "counts": [len(retrieved_memories)],
                "scores": [avg_score],
                "timestamps": [datetime.now()],
                "queries": [query]
            }

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get memory retrieval statistics.
        
        Returns:
            Dictionary with retrieval statistics
        """
        if hasattr(self, "_retrieval_stats"):
            return self._retrieval_stats
        return {"counts": [], "scores": [], "timestamps": [], "queries": []}

    def launch_dashboard(self, port: int = 8050) -> Any:
        """
        Launch the development dashboard.
        
        Args:
            port: Port to run the dashboard on
            
        Returns:
            Dashboard instance
        """
        try:
            from lmm.visualization.dashboard import DevelopmentDashboard
            dashboard = DevelopmentDashboard(lmm_instance=self, port=port)
            dashboard.start_background()
            logger.info(f"Launched dashboard on port {port}")
            return dashboard
        except ImportError as e:
            logger.error(f"Failed to launch dashboard: {e}")
            print(f"Error: Failed to launch dashboard - {e}")
            return None

def main():
    """Main entry point."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Large Mind Model")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stage", help="Set initial developmental stage")
    parser.add_argument("--simulate", type=int, help="Run development simulation with N iterations")
    parser.add_argument("--dashboard", action="store_true", help="Launch development dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Port for the dashboard")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize LMM
    config_dict = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    
    lmm = LargeMindsModel(config_dict)
    
    # Set stage if provided
    if args.stage:
        lmm.set_developmental_stage(args.stage)
    
    # Run simulation if requested
    if args.simulate:
        results = lmm.simulate_development(args.simulate)
        print(f"Simulation results: {results}")
        return
    
    # Launch dashboard if requested
    dashboard = None
    if args.dashboard:
        dashboard = lmm.launch_dashboard(port=args.dashboard_port)
        print(f"Dashboard launched on port {args.dashboard_port}. Access at http://localhost:{args.dashboard_port}")
        # Add a small sleep to ensure the dashboard has time to start
        time.sleep(1)
    
    # Run in interactive mode
    if args.interactive:
        print("LMM Interactive Mode. Type 'exit' to quit.")
        print("Special commands:")
        print(" - 'status': Get development status")
        print(" - 'memory': Get memory status")
        print(" - 'recall <query>': Recall memories")
        print(" - 'introspect': Get introspection")
        print(" - 'modules': Get mind modules status")
        print(" - 'consolidate': Force memory consolidation")
        print(" - 'simulate <N>': Run development simulation")
        print(" - 'stage <stage>': Set developmental stage")
        print(" - 'dashboard': Launch visualization dashboard")
        
        while True:
            try:
                message = input("\nYou: ")
                
                if message.lower() == 'exit':
                    break
                
                # Handle special commands
                if message.lower() == 'status':
                    status = lmm.get_development_status()
                    print("\nDevelopment Status:")
                    print(f"Stage: {status['current_stage']}")
                    print(f"Progress: {status['stage_progress']:.2f}")
                    print(f"Overall Progress: {status['overall_progress']:.2f}")
                    print(f"Interactions: {status['interaction_count']}")
                    print("\nBrain Development:")
                    for key, value in status['brain_development'].items():
                        print(f"- {key}: {value:.3f}")
                    continue
                
                if message.lower() == 'memory':
                    memory_status = lmm.get_memory_status()
                    stats = memory_status.get("memory_stats", {})
                    working = memory_status.get("working_memory", [])
                    
                    print("\nMemory Status:")
                    print(f"Total memories: {stats.get('total_memories', 0)}")
                    print(f"Working memory: {len(working)}/{stats.get('working_memory_capacity', 0)} items")
                    
                    strength_dist = stats.get("strength_distribution", {})
                    print("\nMemory Strength Distribution:")
                    for category, count in strength_dist.items():
                        print(f"- {category}: {count}")
                        
                    print("\nWorking Memory Contents:")
                    for memory in working:
                        print(f"- {memory.get('content', '')[:50]}...")
                    continue
                
                if message.lower().startswith('recall '):
                    query = message[7:]
                    memories = lmm.recall_memories(query, limit=3)
                    
                    print(f"\nMemories related to '{query}':")
                    for memory in memories:
                        print(f"\n- {memory['content']}")
                        print(f"  Type: {memory['type']}, Score: {memory.get('retrieval_score', 0):.2f}")
                        if memory.get("reconstructed"):
                            print(f"  Note: This memory may be partially reconstructed. Confidence: {memory.get('confidence', 0):.2f}")
                    continue
                
                if message.lower() == 'introspect':
                    print("\n" + lmm.get_introspection())
                    continue
                
                if message.lower() == 'modules':
                    modules = lmm.get_mind_modules_status()
                    
                    print("\nMind Modules Status:")
                    for name, module in modules.items():
                        print(f"\n{name.upper()} Module: {module.get('status', 'unknown')}")
                        if name == 'memory':
                            counts = module.get('memory_counts', {})
                            print(f"  Total memories: {counts.get('total', 0)}")
                            print(f"  Working memory: {module.get('working_memory', {}).get('usage', 0)}/{module.get('working_memory', {}).get('capacity', 0)}")
                    continue
                
                if message.lower() == 'consolidate':
                    result = lmm.memory_module.process({
                        "operation": "consolidate",
                        "parameters": {"force": True}
                    })
                    
                    print(f"\nConsolidated {result.get('consolidated_count', 0)} memories.")
                    continue
                
                if message.lower().startswith('simulate '):
                    try:
                        iterations = int(message.split()[1])
                        results = lmm.simulate_development(iterations)
                        
                        print(f"\nSimulation completed with {iterations} iterations.")
                        print(f"Learning progress: {results.get('learning_progress', 0):.3f}")
                        print(f"Memories consolidated: {results.get('memory_consolidation', {}).get('consolidated_count', 0)}")
                    except (ValueError, IndexError):
                        print("Invalid simulate command. Use 'simulate <number>'.")
                    continue
                
                if message.lower().startswith('stage '):
                    stage = message[6:].strip()
                    try:
                        lmm.set_developmental_stage(stage)
                        print(f"\nSet developmental stage to {stage}.")
                    except ValueError:
                        print(f"Invalid stage: {stage}")
                    continue
                
                if message.lower() == 'dashboard':
                    if dashboard:
                        print("\nDashboard is already running")
                    else:
                        dashboard = lmm.launch_dashboard()
                        print(f"\nLaunched dashboard. Access at http://localhost:8050")
                    continue
                
                # Normal interaction
                response = lmm.interact(message)
                print(f"\nLMM: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Non-interactive mode would define an API or other interface
        pass

if __name__ == "__main__":
    main() 