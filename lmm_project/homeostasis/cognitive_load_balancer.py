from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import logging
import math
from collections import defaultdict, deque

from lmm_project.core.message import Message
from lmm_project.core.event_bus import EventBus
from lmm_project.core.types import DevelopmentalStage, StateDict, ModuleType
from .models import HomeostaticSystem, HomeostaticNeedType, HomeostaticResponse, NeedState

logger = logging.getLogger(__name__)

class CognitiveLoadBalancer:
    """
    Manages the distribution of cognitive processing resources.
    
    The Cognitive Load Balancer:
    - Tracks resource usage across modules
    - Prevents overload conditions
    - Prioritizes processing tasks
    - Adjusts resource allocation based on priorities
    - Implements working memory limitations
    
    Cognitive load is analogous to the processing capacity of the mind.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        initial_capacity: float = 0.3,
        working_memory_slots: int = 4,
        processing_threshold: float = 0.8
    ):
        self.event_bus = event_bus
        self.homeostatic_system = HomeostaticSystem()
        self.homeostatic_system.initialize_needs()
        
        # Initialize cognitive load state
        cognitive_load_need = self.homeostatic_system.needs.get(HomeostaticNeedType.COGNITIVE_LOAD)
        if cognitive_load_need:
            cognitive_load_need.current_value = initial_capacity
            cognitive_load_need.last_updated = datetime.now()
        
        # Cognitive capacity parameters
        self.working_memory_slots = working_memory_slots
        self.processing_threshold = processing_threshold
        self.last_update_time = datetime.now()
        
        # Resource tracking
        self.module_load: Dict[str, float] = {}
        self.task_priorities: Dict[str, int] = {}
        self.working_memory_items: List[Dict[str, Any]] = []
        self.processing_queue: deque = deque(maxlen=20)
        self.active_processes: Set[str] = set()
        
        # Development-related parameters
        self.capacity_growth_rate = 0.05  # Increase in capacity per development unit
        self.complexity_tolerance = 0.3   # Ability to handle complex processing
        
        # Recent task history (for detecting patterns)
        self.task_history: List[Dict[str, Any]] = []
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register handlers for cognitive load related events"""
        self.event_bus.subscribe("module_processing_request", self._handle_processing_request)
        self.event_bus.subscribe("working_memory_update", self._handle_working_memory_update)
        self.event_bus.subscribe("system_cycle", self._handle_system_cycle)
        self.event_bus.subscribe("development_update", self._handle_development_update)
        self.event_bus.subscribe("energy_state_update", self._handle_energy_update)
    
    def _handle_processing_request(self, message: Message):
        """Handle module processing requests to track and allocate resources"""
        module_name = message.content.get("module_name", "unknown")
        resource_demand = message.content.get("resource_demand", 0.1)
        priority = message.content.get("priority", 1)
        task_id = message.content.get("task_id", f"task_{len(self.task_history) + 1}")
        concurrent = message.content.get("concurrent", False)
        
        # Record the task request
        task_info = {
            "task_id": task_id,
            "module": module_name,
            "demand": resource_demand,
            "priority": priority,
            "timestamp": datetime.now(),
            "status": "pending"
        }
        self.task_history.append(task_info)
        
        # Check if we can process this request
        current_load = self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value
        
        # If concurrent processing is not allowed, check if there are active processes
        if not concurrent and self.active_processes and module_name not in self.active_processes:
            # Queue for later processing
            self.processing_queue.append(task_info)
            
            queue_message = Message(
                sender="cognitive_load_balancer",
                message_type="processing_queued",
                content={
                    "task_id": task_id,
                    "reason": "Non-concurrent task with existing processes",
                    "queue_position": len(self.processing_queue),
                    "estimated_wait": len(self.processing_queue) * 2.0  # Rough estimate in seconds
                }
            )
            self.event_bus.publish(queue_message)
            return
        
        # Check if adding this would exceed capacity
        if current_load + resource_demand > self.processing_threshold:
            # System is overloaded
            if priority > 3:  # High priority tasks still get processed
                pass  # Continue processing
            else:
                # Queue for later processing
                self.processing_queue.append(task_info)
                
                overload_message = Message(
                    sender="cognitive_load_balancer",
                    message_type="cognitive_overload",
                    content={
                        "current_load": current_load,
                        "threshold": self.processing_threshold,
                        "request_demand": resource_demand,
                        "queued_tasks": len(self.processing_queue)
                    },
                    priority=4
                )
                self.event_bus.publish(overload_message)
                return
        
        # Update cognitive load
        self.homeostatic_system.update_need(
            HomeostaticNeedType.COGNITIVE_LOAD,
            resource_demand,
            f"Processing request from {module_name}"
        )
        
        # Track module load
        self.module_load[module_name] = self.module_load.get(module_name, 0) + resource_demand
        self.task_priorities[task_id] = priority
        self.active_processes.add(module_name)
        
        # Acknowledge processing
        task_info["status"] = "processing"
        ack_message = Message(
            sender="cognitive_load_balancer",
            message_type="processing_allocated",
            content={
                "task_id": task_id,
                "allocated_resources": resource_demand,
                "current_system_load": self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value
            }
        )
        self.event_bus.publish(ack_message)
        
        # Check if we're nearing capacity threshold
        if current_load + resource_demand > self.processing_threshold * 0.9:
            self._signal_approaching_capacity()
    
    def _handle_working_memory_update(self, message: Message):
        """Handle working memory updates to track capacity"""
        operation = message.content.get("operation", "add")
        item = message.content.get("item", {})
        item_id = item.get("id", str(len(self.working_memory_items) + 1))
        
        if operation == "add":
            # Check if we're at capacity
            if len(self.working_memory_items) >= self.working_memory_slots:
                # Need to remove an item - least recently used or lowest priority
                self._evict_working_memory_item()
            
            # Add the new item
            item["added_at"] = datetime.now()
            item["last_accessed"] = datetime.now()
            item["access_count"] = 1
            self.working_memory_items.append(item)
            
            # Update cognitive load (working memory has a cost)
            self.homeostatic_system.update_need(
                HomeostaticNeedType.COGNITIVE_LOAD,
                0.05,  # Small increase for adding to working memory
                "Working memory item added"
            )
            
        elif operation == "remove":
            # Find and remove the item
            self.working_memory_items = [i for i in self.working_memory_items if i.get("id") != item_id]
            
            # Update cognitive load (freeing up working memory)
            self.homeostatic_system.update_need(
                HomeostaticNeedType.COGNITIVE_LOAD,
                -0.05,  # Small decrease for removing from working memory
                "Working memory item removed"
            )
            
        elif operation == "access":
            # Update access time and count for the item
            for i in self.working_memory_items:
                if i.get("id") == item_id:
                    i["last_accessed"] = datetime.now()
                    i["access_count"] = i.get("access_count", 0) + 1
                    break
        
        # Publish current working memory state
        self._publish_working_memory_state()
    
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle events to update cognitive load naturally"""
        # Calculate time since last update
        now = datetime.now()
        time_delta = (now - self.last_update_time).total_seconds()
        
        # Natural decay of cognitive load (processes completing)
        if time_delta > 1.0:
            decay_amount = 0.02 * time_delta / 5.0  # Slow natural decay
            
            # Only decay if we have a positive cognitive load
            if self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value > 0.1:
                self.homeostatic_system.update_need(
                    HomeostaticNeedType.COGNITIVE_LOAD,
                    -decay_amount,
                    "Natural cognitive load decay"
                )
                self.last_update_time = now
            
            # Process queued tasks if capacity available
            self._process_queued_tasks()
            
            # Publish current cognitive load state
            self._publish_cognitive_load_state()
    
    def _handle_development_update(self, message: Message):
        """Adapt cognitive capacity based on developmental stage"""
        development_level = message.content.get("development_level", 0.0)
        
        # Update homeostatic setpoints based on development
        self.homeostatic_system.adapt_to_development(development_level)
        
        # Adjust cognitive parameters based on development
        # Working memory increases with development
        self.working_memory_slots = 3 + math.floor(development_level * 4)  # 3-7 slots
        
        # Processing capacity increases with development
        self.complexity_tolerance = 0.3 + (development_level * 0.5)  # 0.3-0.8
        
        logger.info(
            f"Cognitive load balancer adapted to development level {development_level:.2f}: "
            f"WM slots={self.working_memory_slots}, complexity={self.complexity_tolerance:.2f}"
        )
    
    def _handle_energy_update(self, message: Message):
        """Adjust cognitive processing based on energy levels"""
        energy_level = message.content.get("current_energy", 0.5)
        is_deficient = message.content.get("is_deficient", False)
        
        if is_deficient:
            # Reduce processing capacity when energy is low
            old_threshold = self.processing_threshold
            self.processing_threshold = 0.6 * (energy_level + 0.4)
            
            logger.info(
                f"Cognitive capacity reduced due to low energy: {old_threshold:.2f} → {self.processing_threshold:.2f}"
            )
            
            # If current load exceeds new threshold, need to shed some load
            current_load = self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value
            if current_load > self.processing_threshold:
                self._shed_cognitive_load()
        else:
            # Restore normal processing capacity
            self.processing_threshold = 0.8
    
    def _process_queued_tasks(self):
        """Process any queued tasks if capacity is available"""
        current_load = self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value
        available_capacity = self.processing_threshold - current_load
        
        # Process queued tasks in priority order
        if available_capacity > 0.1 and self.processing_queue:
            # Sort by priority
            tasks = list(self.processing_queue)
            tasks.sort(key=lambda x: x.get("priority", 0), reverse=True)
            
            for task in tasks:
                demand = task.get("demand", 0.1)
                
                if demand <= available_capacity:
                    # We can process this task
                    self.processing_queue.remove(task)
                    
                    # Update cognitive load
                    self.homeostatic_system.update_need(
                        HomeostaticNeedType.COGNITIVE_LOAD,
                        demand,
                        f"Dequeued task from {task.get('module', 'unknown')}"
                    )
                    
                    # Track module load
                    module_name = task.get("module", "unknown")
                    self.module_load[module_name] = self.module_load.get(module_name, 0) + demand
                    
                    # Update status and notify
                    task["status"] = "processing"
                    self.active_processes.add(module_name)
                    
                    dequeue_message = Message(
                        sender="cognitive_load_balancer",
                        message_type="processing_started",
                        content={
                            "task_id": task.get("task_id", "unknown"),
                            "allocated_resources": demand,
                            "wait_time": (datetime.now() - task.get("timestamp", datetime.now())).total_seconds()
                        }
                    )
                    self.event_bus.publish(dequeue_message)
                    
                    # Update available capacity
                    available_capacity -= demand
                    
                    # Only process one task per cycle to avoid sudden load spikes
                    break
    
    def _shed_cognitive_load(self):
        """Shed cognitive load when system is overloaded"""
        # Find low priority active processes
        low_priority_tasks = [
            task_id for task_id, priority in self.task_priorities.items() 
            if priority < 3
        ]
        
        if not low_priority_tasks:
            return
            
        # Create message to terminate low priority processing
        terminate_message = Message(
            sender="cognitive_load_balancer",
            message_type="terminate_processing",
            content={
                "reason": "Energy conservation needed",
                "tasks": low_priority_tasks[:2]  # Terminate up to 2 low priority tasks
            },
            priority=4
        )
        self.event_bus.publish(terminate_message)
        
        # Reduce recorded load (actual reduction will happen when modules respond)
        self.homeostatic_system.update_need(
            HomeostaticNeedType.COGNITIVE_LOAD,
            -0.15,  # Approximate load reduction
            "Shedding cognitive load due to capacity constraints"
        )
        
        logger.warning("Terminating low priority processing to reduce cognitive load")
    
    def _evict_working_memory_item(self):
        """Evict an item from working memory when at capacity"""
        if not self.working_memory_items:
            return
            
        # Calculate a score for each item based on recency and access count
        for item in self.working_memory_items:
            # Recency score (0-1, higher is more recent)
            time_since_access = (datetime.now() - item.get("last_accessed", datetime.now())).total_seconds()
            recency_score = math.exp(-time_since_access / 60.0)  # Exponential decay with 1-minute half-life
            
            # Access count score (0-1, higher is more frequently accessed)
            access_count = item.get("access_count", 1)
            access_score = min(1.0, access_count / 10.0)
            
            # Importance score (if available)
            importance = item.get("importance", 0.5)
            
            # Combined score (weighted)
            item["retention_score"] = (0.4 * recency_score) + (0.4 * access_score) + (0.2 * importance)
        
        # Find the item with the lowest score
        self.working_memory_items.sort(key=lambda x: x.get("retention_score", 0))
        evicted_item = self.working_memory_items.pop(0)
        
        # Notify about eviction
        eviction_message = Message(
            sender="cognitive_load_balancer",
            message_type="working_memory_eviction",
            content={
                "item_id": evicted_item.get("id", "unknown"),
                "item_content": evicted_item.get("content", {}),
                "reason": "Working memory capacity exceeded",
                "retention_score": evicted_item.get("retention_score", 0)
            }
        )
        self.event_bus.publish(eviction_message)
    
    def _signal_approaching_capacity(self):
        """Signal that the system is approaching cognitive capacity"""
        warning_message = Message(
            sender="cognitive_load_balancer",
            message_type="approaching_capacity",
            content={
                "current_load": self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value,
                "threshold": self.processing_threshold,
                "high_load_modules": self._get_high_load_modules()
            }
        )
        self.event_bus.publish(warning_message)
    
    def _get_high_load_modules(self) -> List[Tuple[str, float]]:
        """Identify modules with highest cognitive load"""
        return sorted(
            [(module, load) for module, load in self.module_load.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 consumers
    
    def _publish_cognitive_load_state(self):
        """Publish current cognitive load state to the event bus"""
        load_need = self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD]
        load_message = Message(
            sender="cognitive_load_balancer",
            message_type="cognitive_load_update",
            content={
                "current_load": load_need.current_value,
                "threshold": self.processing_threshold,
                "utilization": load_need.current_value / self.processing_threshold if self.processing_threshold > 0 else 1.0,
                "module_loads": self.module_load,
                "queue_depth": len(self.processing_queue),
                "active_processes": list(self.active_processes)
            }
        )
        self.event_bus.publish(load_message)
    
    def _publish_working_memory_state(self):
        """Publish current working memory state to the event bus"""
        wm_message = Message(
            sender="cognitive_load_balancer",
            message_type="working_memory_state",
            content={
                "items": [i.get("id", "unknown") for i in self.working_memory_items],
                "used_slots": len(self.working_memory_items),
                "total_slots": self.working_memory_slots,
                "utilization": len(self.working_memory_items) / self.working_memory_slots if self.working_memory_slots > 0 else 1.0
            }
        )
        self.event_bus.publish(wm_message)
    
    def release_resources(self, task_id: str, module_name: str, amount: float) -> None:
        """
        Release cognitive resources when a task completes
        
        Arguments:
            task_id: The ID of the completed task
            module_name: The module that was using the resources
            amount: The amount of resources to release
        """
        # Update cognitive load
        self.homeostatic_system.update_need(
            HomeostaticNeedType.COGNITIVE_LOAD,
            -amount,
            f"Task completion in {module_name}"
        )
        
        # Update module load
        if module_name in self.module_load:
            self.module_load[module_name] = max(0, self.module_load.get(module_name, 0) - amount)
            
            # If module has no load, remove from active processes
            if self.module_load[module_name] <= 0:
                self.active_processes.discard(module_name)
        
        # Remove task priority tracking
        if task_id in self.task_priorities:
            del self.task_priorities[task_id]
        
        # Update task history
        for task in self.task_history:
            if task.get("task_id") == task_id:
                task["status"] = "completed"
                task["completion_time"] = datetime.now()
                break
                
        # Process queued tasks if we freed up capacity
        self._process_queued_tasks()
        
        # Publish updated state
        self._publish_cognitive_load_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the cognitive load balancer"""
        load_need = self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD]
        return {
            "cognitive_load": load_need.current_value,
            "processing_threshold": self.processing_threshold,
            "working_memory_slots": self.working_memory_slots,
            "working_memory_usage": len(self.working_memory_items),
            "working_memory_items": [i.get("id", "unknown") for i in self.working_memory_items],
            "queue_depth": len(self.processing_queue),
            "active_processes": list(self.active_processes),
            "module_loads": self.module_load,
            "complexity_tolerance": self.complexity_tolerance
        }
    
    def load_state(self, state_dict: StateDict) -> None:
        """Load state from the provided state dictionary"""
        if "cognitive_load" in state_dict:
            self.homeostatic_system.update_need(
                HomeostaticNeedType.COGNITIVE_LOAD,
                state_dict["cognitive_load"] - 
                self.homeostatic_system.needs[HomeostaticNeedType.COGNITIVE_LOAD].current_value,
                "State loaded"
            )
            
        if "processing_threshold" in state_dict:
            self.processing_threshold = state_dict["processing_threshold"]
            
        if "working_memory_slots" in state_dict:
            self.working_memory_slots = state_dict["working_memory_slots"]
            
        if "module_loads" in state_dict:
            self.module_load = state_dict["module_loads"]
            
        if "active_processes" in state_dict:
            self.active_processes = set(state_dict["active_processes"])
            
        if "complexity_tolerance" in state_dict:
            self.complexity_tolerance = state_dict["complexity_tolerance"]
