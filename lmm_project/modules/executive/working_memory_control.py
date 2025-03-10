# TODO: Implement the WorkingMemoryControl class to manage working memory contents
# This component should be able to:
# - Maintain information in an active state
# - Update working memory contents as needed
# - Protect contents from interference
# - Manipulate and transform held information

# TODO: Implement developmental progression in working memory control:
# - Very limited capacity and duration in early stages
# - Gradual increase in capacity during childhood
# - Improved manipulation abilities in adolescence
# - Strategic working memory management in adulthood

# TODO: Create mechanisms for:
# - Maintenance: Keep information active through rehearsal
# - Updating: Replace old information with new when appropriate
# - Binding: Associate multiple pieces of information together
# - Manipulation: Transform or reorganize held information

# TODO: Implement capacity limitations:
# - Limit on number of items that can be held simultaneously
# - Limit on complexity of items based on developmental level
# - Trade-offs between maintenance and manipulation
# - Interference effects between similar items

# TODO: Connect to attention and consciousness systems
# Working memory should be influenced by attentional focus
# and should feed information to conscious awareness

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.executive.models import WorkingMemoryItem, WorkingMemoryState, ExecutiveNeuralState
from lmm_project.modules.executive.neural_net import WorkingMemoryNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class WorkingMemoryControl(BaseModule):
    """
    Manages the contents of working memory
    
    This module controls what information is maintained in an active state,
    updated, protected from interference, and manipulated.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic maintenance",
        0.2: "Improved capacity",
        0.4: "Active updating",
        0.6: "Information manipulation",
        0.8: "Strategic memory allocation",
        1.0: "Sophisticated working memory control"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the working memory control module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="working_memory_control", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device
        self.device = get_device()
        
        # Initialize neural network
        self.memory_network = WorkingMemoryNetwork(
            item_dim=64,
            control_dim=32,
            hidden_dim=128,
            capacity=7  # Maximum theoretical capacity
        ).to(self.device)
        
        # Set development level for network
        self.memory_network.set_development_level(development_level)
        
        # Create neural state for tracking
        self.neural_state = ExecutiveNeuralState()
        self.neural_state.working_memory_development = development_level
        
        # Initialize working memory state
        self.memory_state = WorkingMemoryState(
            capacity=2 + int(5 * development_level),  # Capacity increases with development
            capacity_utilization=0.0,
            last_updated=datetime.now()
        )
        
        # Last decay update timestamp
        self.last_decay_update = time.time()
        
        # Working memory parameters
        self.params = {
            "activation_decay_rate": 0.1,  # How quickly items decay
            "retrieval_threshold": 0.2,  # Minimum activation to retrieve
            "interference_factor": 0.3,  # How much similar items interfere
            "rehearsal_boost": 0.5,  # Activation boost from rehearsal
            "manipulation_cost": 0.2  # Activation cost of manipulation
        }
        
        # Update parameters based on development
        self._adjust_parameters_for_development()
        
        logger.info(f"Working memory control module initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust working memory parameters based on developmental level"""
        if self.development_level < 0.2:
            # Very basic working memory at early stages
            self.params.update({
                "activation_decay_rate": 0.2,  # Faster decay
                "retrieval_threshold": 0.3,  # Higher threshold (harder to retrieve)
                "interference_factor": 0.5,  # More interference
                "rehearsal_boost": 0.3,  # Smaller boost from rehearsal
                "manipulation_cost": 0.4  # Higher cost for manipulation
            })
            
            # Update capacity in state
            self.memory_state.capacity = max(1, 2 + int(5 * self.development_level))
            
        elif self.development_level < 0.4:
            # Developing basic capacity
            self.params.update({
                "activation_decay_rate": 0.15,
                "retrieval_threshold": 0.25,
                "interference_factor": 0.4,
                "rehearsal_boost": 0.4,
                "manipulation_cost": 0.3
            })
            
            # Update capacity in state
            self.memory_state.capacity = max(2, 2 + int(5 * self.development_level))
            
        elif self.development_level < 0.6:
            # Improved updating capabilities
            self.params.update({
                "activation_decay_rate": 0.1,
                "retrieval_threshold": 0.2,
                "interference_factor": 0.3,
                "rehearsal_boost": 0.5,
                "manipulation_cost": 0.25
            })
            
            # Update capacity in state
            self.memory_state.capacity = max(3, 2 + int(5 * self.development_level))
            
        elif self.development_level < 0.8:
            # Developing manipulation abilities
            self.params.update({
                "activation_decay_rate": 0.08,
                "retrieval_threshold": 0.15,
                "interference_factor": 0.2,
                "rehearsal_boost": 0.6,
                "manipulation_cost": 0.2
            })
            
            # Update capacity in state
            self.memory_state.capacity = max(4, 2 + int(5 * self.development_level))
            
        else:
            # Strategic memory control
            self.params.update({
                "activation_decay_rate": 0.05,
                "retrieval_threshold": 0.1,
                "interference_factor": 0.1,
                "rehearsal_boost": 0.7,
                "manipulation_cost": 0.1
            })
            
            # Update capacity in state
            self.memory_state.capacity = max(6, 2 + int(5 * self.development_level))
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to control working memory
        
        Args:
            input_data: Dictionary containing working memory operations
                Required keys:
                - 'operation': The operation to perform ('store', 'retrieve', 'update', 'clear', 'query')
                For 'store' operation:
                - 'content': The content to store
                - 'content_type': Type of content (visual, verbal, spatial, etc.)
                For 'retrieve' operation:
                - 'item_id' or 'query': How to find the item (direct ID or search query)
                For 'update' operation:
                - 'item_id': ID of item to update
                - 'content': New content
                For 'clear' operation:
                - 'item_id' (optional): Specific item to clear, otherwise clear all
            
        Returns:
            Dictionary with the results of working memory control
        """
        # Update activation decay
        self._update_activations()
        
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        operation = input_data.get("operation", "query")
        
        # Different operations based on the request
        if operation == "store":
            return self._store_item(input_data, process_id)
        elif operation == "retrieve":
            return self._retrieve_item(input_data, process_id)
        elif operation == "update":
            return self._update_item(input_data, process_id)
        elif operation == "manipulate":
            return self._manipulate_item(input_data, process_id)
        elif operation == "clear":
            return self._clear_items(input_data, process_id)
        elif operation == "query":
            return self._query_memory(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _store_item(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Store a new item in working memory"""
        # Extract required data
        if "content" not in input_data:
            return {"status": "error", "message": "No content provided", "process_id": process_id}
        
        content = input_data.get("content")
        content_type = input_data.get("content_type", "general")
        tags = input_data.get("tags", [])
        
        # Check capacity
        if len(self.memory_state.items) >= self.memory_state.capacity:
            if self.development_level < 0.6:
                # At lower development, simply fail when capacity is reached
                return {
                    "status": "error",
                    "message": "Working memory capacity reached",
                    "capacity": self.memory_state.capacity,
                    "current_items": len(self.memory_state.items),
                    "process_id": process_id
                }
            else:
                # At higher development, remove least active item
                least_active_id = min(
                    self.memory_state.items.items(), 
                    key=lambda x: x[1].activation
                )[0]
                
                # Remove item
                del self.memory_state.items[least_active_id]
                
                # Log the replacement
                logger.info(f"Replaced least active item {least_active_id} to make room for new item")
        
        # Convert content to tensors for neural processing
        content_features = self._extract_features(content)
        
        # Create dummy control tensor (for store operation)
        control_features = torch.zeros((1, 32), dtype=torch.float32)
        
        # Process through neural network
        with torch.no_grad():
            store_result = self.memory_network(
                operation='store',
                items=content_features.to(self.device),
                control=control_features.to(self.device)
            )
        
        # Create new memory item
        item_id = str(uuid.uuid4())
        new_item = WorkingMemoryItem(
            item_id=item_id,
            content=content,
            content_type=content_type,
            activation=1.0,  # Start with maximum activation
            tags=tags,
            creation_time=datetime.now(),
            last_access=datetime.now(),
            access_count=1
        )
        
        # Store in working memory
        self.memory_state.items[item_id] = new_item
        
        # Update focus of attention
        self.memory_state.focus_of_attention = item_id
        self.memory_state.last_operation = "store"
        
        # Update utilization
        self.memory_state.capacity_utilization = len(self.memory_state.items) / self.memory_state.capacity
        
        # Update timestamp
        self.memory_state.last_updated = datetime.now()
        
        # Record activation in neural state
        self.neural_state.add_activation('working_memory', {
            'operation': 'store',
            'content_type': content_type,
            'success': store_result.get('store_success', torch.tensor([1.0])).item()
        })
        
        return {
            "status": "success",
            "item_id": item_id,
            "operation": "store",
            "memory_item": new_item.dict(),
            "capacity_utilization": self.memory_state.capacity_utilization,
            "process_id": process_id
        }
    
    def _retrieve_item(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Retrieve an item from working memory"""
        # Extract retrieval parameters
        item_id = input_data.get("item_id")
        query = input_data.get("query")
        
        if not item_id and not query:
            return {"status": "error", "message": "No item_id or query provided", "process_id": process_id}
        
        retrieved_item = None
        retrieval_method = "direct"
        
        if item_id:
            # Direct retrieval by ID
            if item_id in self.memory_state.items:
                retrieved_item = self.memory_state.items[item_id]
                
                # Check activation threshold
                if retrieved_item.activation < self.params["retrieval_threshold"]:
                    return {
                        "status": "forgotten",
                        "message": f"Item activation below threshold: {retrieved_item.activation:.2f}",
                        "item_id": item_id,
                        "activation": retrieved_item.activation,
                        "threshold": self.params["retrieval_threshold"],
                        "process_id": process_id
                    }
            else:
                return {
                    "status": "not_found",
                    "message": f"Item with ID {item_id} not found",
                    "process_id": process_id
                }
        else:
            # Query-based retrieval
            retrieval_method = "query"
            query_features = self._extract_features(query)
            
            # Create control tensor from query
            control_features = torch.zeros((1, 32), dtype=torch.float32)
            for i, val in enumerate(query_features.squeeze().tolist()):
                if i < 32:
                    control_features[0, i] = val
            
            # Process through neural network
            with torch.no_grad():
                retrieve_result = self.memory_network(
                    operation='retrieve',
                    control=control_features.to(self.device)
                )
            
            # Find the best matching item
            if self.memory_state.items:
                # Convert query to features
                query_features_np = query_features.cpu().numpy()
                
                # For each item, calculate similarity to query
                best_score = -1
                best_item_id = None
                
                for item_id, item in self.memory_state.items.items():
                    # Skip items below threshold
                    if item.activation < self.params["retrieval_threshold"]:
                        continue
                        
                    # Calculate similarity (simple cosine)
                    item_features = self._extract_features(item.content).cpu().numpy()
                    similarity = np.sum(query_features_np * item_features) / (
                        np.sqrt(np.sum(query_features_np ** 2)) * 
                        np.sqrt(np.sum(item_features ** 2))
                    )
                    
                    # Apply activation as a weight
                    weighted_similarity = similarity * item.activation
                    
                    if weighted_similarity > best_score:
                        best_score = weighted_similarity
                        best_item_id = item_id
                
                if best_item_id:
                    retrieved_item = self.memory_state.items[best_item_id]
                    item_id = best_item_id
                    
                    # Record the query match in neural state
                    self.neural_state.add_activation('working_memory', {
                        'operation': 'retrieve',
                        'method': 'query',
                        'similarity': best_score
                    })
            
            if not retrieved_item:
                return {
                    "status": "not_found",
                    "message": "No items matching query",
                    "query": query,
                    "process_id": process_id
                }
        
        # Update item data on successful retrieval
        retrieved_item.access_count += 1
        retrieved_item.last_access = datetime.now()
        
        # Boost activation from retrieval
        retrieved_item.activation = min(1.0, retrieved_item.activation + self.params["rehearsal_boost"])
        
        # Update focus of attention
        self.memory_state.focus_of_attention = item_id
        self.memory_state.last_operation = "retrieve"
        
        # Update timestamp
        self.memory_state.last_updated = datetime.now()
        
        # Record activation in neural state
        self.neural_state.add_activation('working_memory', {
            'operation': 'retrieve',
            'method': retrieval_method,
            'item_id': item_id,
            'content_type': retrieved_item.content_type,
            'activation': retrieved_item.activation
        })
        
        return {
            "status": "success",
            "item_id": item_id,
            "operation": "retrieve",
            "memory_item": retrieved_item.dict(),
            "process_id": process_id
        }
    
    def _update_item(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Update an existing item in working memory"""
        # Extract required data
        if "item_id" not in input_data:
            return {"status": "error", "message": "No item_id provided", "process_id": process_id}
        if "content" not in input_data:
            return {"status": "error", "message": "No content provided", "process_id": process_id}
        
        item_id = input_data.get("item_id")
        content = input_data.get("content")
        
        # Check if item exists
        if item_id not in self.memory_state.items:
            return {
                "status": "not_found",
                "message": f"Item with ID {item_id} not found",
                "process_id": process_id
            }
        
        # Get the existing item
        item = self.memory_state.items[item_id]
        
        # Check activation threshold
        if item.activation < self.params["retrieval_threshold"]:
            return {
                "status": "forgotten",
                "message": f"Item activation below threshold: {item.activation:.2f}",
                "item_id": item_id,
                "activation": item.activation,
                "threshold": self.params["retrieval_threshold"],
                "process_id": process_id
            }
        
        # Convert content to tensors for neural processing
        content_features = self._extract_features(content)
        
        # Create control tensor for item ID
        control_features = self._extract_features(item_id)
        control_features_resized = torch.zeros((1, 32), dtype=torch.float32)
        for i, val in enumerate(control_features.squeeze().tolist()):
            if i < 32:
                control_features_resized[0, i] = val
        
        # Process through neural network
        with torch.no_grad():
            update_result = self.memory_network(
                operation='update',
                items=content_features.to(self.device),
                control=control_features_resized.to(self.device)
            )
        
        # Update the item
        old_content = item.content
        item.content = content
        item.access_count += 1
        item.last_access = datetime.now()
        
        # Updating takes some activation resources
        item.activation = max(
            self.params["retrieval_threshold"],
            item.activation - self.params["manipulation_cost"]
        )
        
        # Update focus of attention
        self.memory_state.focus_of_attention = item_id
        self.memory_state.last_operation = "update"
        
        # Update timestamp
        self.memory_state.last_updated = datetime.now()
        
        # Record activation in neural state
        self.neural_state.add_activation('working_memory', {
            'operation': 'update',
            'item_id': item_id,
            'content_type': item.content_type,
            'activation': item.activation
        })
        
        return {
            "status": "success",
            "item_id": item_id,
            "operation": "update",
            "previous_content": old_content,
            "memory_item": item.dict(),
            "process_id": process_id
        }
    
    def _manipulate_item(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Transform or manipulate content in working memory"""
        # Only available at higher development levels
        if self.development_level < 0.6:
            return {
                "status": "undeveloped",
                "message": "Manipulation operations require higher development",
                "development_level": self.development_level,
                "required_level": 0.6,
                "process_id": process_id
            }
            
        # Extract required data
        if "item_id" not in input_data:
            return {"status": "error", "message": "No item_id provided", "process_id": process_id}
        if "operation_type" not in input_data:
            return {"status": "error", "message": "No operation_type provided", "process_id": process_id}
        
        item_id = input_data.get("item_id")
        operation_type = input_data.get("operation_type")
        parameters = input_data.get("parameters", {})
        
        # Check if item exists
        if item_id not in self.memory_state.items:
            return {
                "status": "not_found",
                "message": f"Item with ID {item_id} not found",
                "process_id": process_id
            }
        
        # Get the existing item
        item = self.memory_state.items[item_id]
        
        # Check activation threshold
        if item.activation < self.params["retrieval_threshold"]:
            return {
                "status": "forgotten",
                "message": f"Item activation below threshold: {item.activation:.2f}",
                "item_id": item_id,
                "activation": item.activation,
                "threshold": self.params["retrieval_threshold"],
                "process_id": process_id
            }
        
        # Manipulate based on operation type
        old_content = item.content
        manipulation_result = None
        
        try:
            if operation_type == "transform":
                # Apply a transformation to the content
                transform_type = parameters.get("transform_type", "")
                
                if isinstance(item.content, str):
                    if transform_type == "uppercase":
                        manipulation_result = item.content.upper()
                    elif transform_type == "lowercase":
                        manipulation_result = item.content.lower()
                    elif transform_type == "reverse":
                        manipulation_result = item.content[::-1]
                    else:
                        return {"status": "error", "message": f"Unknown transform_type: {transform_type}", "process_id": process_id}
                        
                elif isinstance(item.content, (list, tuple)):
                    if transform_type == "reverse":
                        manipulation_result = list(reversed(item.content))
                    elif transform_type == "sort":
                        manipulation_result = sorted(item.content)
                    else:
                        return {"status": "error", "message": f"Unknown transform_type: {transform_type}", "process_id": process_id}
                        
                elif isinstance(item.content, dict):
                    if transform_type == "keys":
                        manipulation_result = list(item.content.keys())
                    elif transform_type == "values":
                        manipulation_result = list(item.content.values())
                    else:
                        return {"status": "error", "message": f"Unknown transform_type: {transform_type}", "process_id": process_id}
                else:
                    return {"status": "error", "message": "Content type cannot be transformed", "process_id": process_id}
                    
            elif operation_type == "combine":
                # Combine with another item
                other_id = parameters.get("other_id")
                
                if not other_id or other_id not in self.memory_state.items:
                    return {"status": "error", "message": f"Other item {other_id} not found", "process_id": process_id}
                    
                other_item = self.memory_state.items[other_id]
                
                # Check other item activation
                if other_item.activation < self.params["retrieval_threshold"]:
                    return {
                        "status": "forgotten",
                        "message": f"Other item activation below threshold: {other_item.activation:.2f}",
                        "item_id": other_id,
                        "process_id": process_id
                    }
                
                # Combine based on content types
                if isinstance(item.content, str) and isinstance(other_item.content, str):
                    manipulation_result = item.content + " " + other_item.content
                    
                elif isinstance(item.content, (list, tuple)) and isinstance(other_item.content, (list, tuple)):
                    manipulation_result = list(item.content) + list(other_item.content)
                    
                elif isinstance(item.content, dict) and isinstance(other_item.content, dict):
                    manipulation_result = {**item.content, **other_item.content}
                    
                else:
                    manipulation_result = [item.content, other_item.content]
                
                # Also update other item's activation and access count
                other_item.access_count += 1
                other_item.last_access = datetime.now()
                other_item.activation = max(
                    self.params["retrieval_threshold"],
                    other_item.activation - self.params["manipulation_cost"]
                )
                
            else:
                return {"status": "error", "message": f"Unknown operation_type: {operation_type}", "process_id": process_id}
                
            # Update the item with manipulation result
            item.content = manipulation_result
            item.access_count += 1
            item.last_access = datetime.now()
            
            # Manipulation takes significant activation resources
            item.activation = max(
                self.params["retrieval_threshold"],
                item.activation - self.params["manipulation_cost"] * 1.5
            )
            
            # Update focus of attention
            self.memory_state.focus_of_attention = item_id
            self.memory_state.last_operation = f"manipulate_{operation_type}"
            
            # Update timestamp
            self.memory_state.last_updated = datetime.now()
            
            # Record activation in neural state
            self.neural_state.add_activation('working_memory', {
                'operation': 'manipulate',
                'sub_operation': operation_type,
                'item_id': item_id,
                'content_type': item.content_type,
                'activation': item.activation
            })
            
            return {
                "status": "success",
                "item_id": item_id,
                "operation": "manipulate",
                "operation_type": operation_type,
                "previous_content": old_content,
                "memory_item": item.dict(),
                "process_id": process_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Manipulation failed: {str(e)}",
                "operation_type": operation_type,
                "process_id": process_id
            }
    
    def _clear_items(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Clear items from working memory"""
        item_id = input_data.get("item_id")
        
        if item_id:
            # Clear specific item
            if item_id in self.memory_state.items:
                deleted_item = self.memory_state.items[item_id]
                del self.memory_state.items[item_id]
                
                # Update focus if needed
                if self.memory_state.focus_of_attention == item_id:
                    self.memory_state.focus_of_attention = None
                
                # Update utilization
                self.memory_state.capacity_utilization = len(self.memory_state.items) / self.memory_state.capacity
                
                # Update timestamp
                self.memory_state.last_updated = datetime.now()
                self.memory_state.last_operation = "clear_item"
                
                # Record activation in neural state
                self.neural_state.add_activation('working_memory', {
                    'operation': 'clear',
                    'item_id': item_id,
                    'content_type': deleted_item.content_type
                })
                
                return {
                    "status": "success",
                    "operation": "clear",
                    "item_id": item_id,
                    "message": f"Item {item_id} cleared",
                    "process_id": process_id
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Item with ID {item_id} not found",
                    "process_id": process_id
                }
        else:
            # Clear all items
            item_count = len(self.memory_state.items)
            self.memory_state.items = {}
            self.memory_state.focus_of_attention = None
            self.memory_state.capacity_utilization = 0.0
            
            # Update timestamp
            self.memory_state.last_updated = datetime.now()
            self.memory_state.last_operation = "clear_all"
            
            # Process through neural network (dummy call for clear operation)
            with torch.no_grad():
                self.memory_network(operation='clear')
            
            # Record activation in neural state
            self.neural_state.add_activation('working_memory', {
                'operation': 'clear_all',
                'item_count': item_count
            })
            
            return {
                "status": "success",
                "operation": "clear",
                "message": f"All {item_count} items cleared",
                "process_id": process_id
            }
    
    def _query_memory(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Query information about working memory"""
        query_type = input_data.get("query_type", "state")
        
        if query_type == "state":
            # Basic state information
            return {
                "status": "success",
                "operation": "query",
                "query_type": "state",
                "capacity": self.memory_state.capacity,
                "capacity_utilization": self.memory_state.capacity_utilization,
                "item_count": len(self.memory_state.items),
                "focus_of_attention": self.memory_state.focus_of_attention,
                "last_operation": self.memory_state.last_operation,
                "process_id": process_id
            }
            
        elif query_type == "items":
            # List all items
            return {
                "status": "success",
                "operation": "query",
                "query_type": "items",
                "items": {id: item.dict() for id, item in self.memory_state.items.items()},
                "item_count": len(self.memory_state.items),
                "process_id": process_id
            }
            
        elif query_type == "activation":
            # Get activation levels
            return {
                "status": "success",
                "operation": "query",
                "query_type": "activation",
                "activations": {id: item.activation for id, item in self.memory_state.items.items()},
                "threshold": self.params["retrieval_threshold"],
                "process_id": process_id
            }
            
        else:
            # Full state
            return {
                "status": "success",
                "operation": "query",
                "working_memory_state": self.memory_state.dict(),
                "working_memory_params": self.params,
                "development_level": self.development_level,
                "process_id": process_id
            }
    
    def _update_activations(self):
        """Update activation decay based on elapsed time"""
        current_time = time.time()
        elapsed_seconds = current_time - self.last_decay_update
        
        if elapsed_seconds > 0.1 and self.memory_state.items:  # Only update if enough time has passed
            # Calculate decay amount
            decay_amount = self.params["activation_decay_rate"] * elapsed_seconds
            
            # Apply decay to all items
            decayed = False
            for item in self.memory_state.items.values():
                old_activation = item.activation
                item.activation = max(0.0, item.activation - decay_amount)
                
                if abs(old_activation - item.activation) > 0.01:
                    decayed = True
            
            # Update timestamp
            self.last_decay_update = current_time
            
            # Update state timestamp if activations changed significantly
            if decayed:
                self.memory_state.last_updated = datetime.now()
    
    def _extract_features(self, data) -> torch.Tensor:
        """
        Extract features from input data for neural processing
        
        Args:
            data: Text, dict, or other data to extract features from
            
        Returns:
            Tensor of features [1, feature_dim]
        """
        # For demonstration, create simple random features
        # In a real implementation, this would use proper feature extraction
        feature_dim = 64
        
        if isinstance(data, str):
            # Seed random generator with hash of string to ensure consistent features
            seed = hash(data) % 10000
            np.random.seed(seed)
            
            # Generate "features" based on the text
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        elif isinstance(data, dict):
            # For dictionary data, use keys and values to generate features
            seed = hash(str(sorted(data.items()))) % 10000
            np.random.seed(seed)
            
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        elif isinstance(data, (list, tuple)):
            # For list/tuple data
            seed = hash(str(data)) % 10000
            np.random.seed(seed)
            
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
            
        else:
            # Default random features
            seed = hash(str(data)) % 10000
            np.random.seed(seed)
            
            features = np.random.randn(feature_dim)
            features = features / np.linalg.norm(features)  # Normalize
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update network development level
        self.memory_network.set_development_level(new_level)
        
        # Update neural state
        self.neural_state.working_memory_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # Adjust parameters based on new development level
        self._adjust_parameters_for_development()
        
        logger.info(f"Working memory control module development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Update activations first
        self._update_activations()
        
        # Get base state from parent
        base_state = super().get_state()
        
        # Add memory-specific state
        memory_state_dict = self.memory_state.dict()
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.working_memory_development,
            "accuracy": self.neural_state.working_memory_accuracy,
            "recent_activations_count": len(self.neural_state.recent_working_memory_activations)
        }
        
        # Combine states
        combined_state = {
            **base_state, 
            **memory_state_dict, 
            "params": self.params,
            "neural_state": neural_state
        }
        
        return combined_state
