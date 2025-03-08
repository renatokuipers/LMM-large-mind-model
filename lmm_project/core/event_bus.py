from typing import Dict, List, Optional, Union, Any, Callable 
from pydantic import BaseModel, Field 
 
class EventBus(BaseModel): 
    subscribers: Dict[str, List[Callable]] = Field(default_factory=dict) 
 
    def publish(self, event_type: str, data: Dict[str, Any]) -
        """Send event to all subscribers""" 
        if event_type in self.subscribers: 
            for callback in self.subscribers[event_type]: 
                callback(data) 
 
    def subscribe(self, event_type: str, callback: Callable) -
        """Register to receive events""" 
        if event_type not in self.subscribers: 
            self.subscribers[event_type] = [] 
        self.subscribers[event_type].append(callback) 
