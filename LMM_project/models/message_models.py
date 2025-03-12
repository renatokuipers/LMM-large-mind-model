from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum, auto

from pydantic import BaseModel, Field, field_validator, computed_field

# TODO: Define MessageType enum for different message categories
#   - COMMAND (module control commands)
#   - DATA (content passing)
#   - QUERY (information requests)
#   - RESPONSE (replies to queries)
#   - EVENT (notifications)

# TODO: Create ModuleAddress model for routing
#   - module_id: str
#   - component: Optional[str]

# TODO: Define BaseMessage model with:
#   - message_id: UUID
#   - timestamp: datetime
#   - source: ModuleAddress
#   - destination: ModuleAddress
#   - message_type: MessageType
#   - trace_id: UUID (for tracking message chains)

# TODO: Implement CommandMessage for module control
#   - command: str
#   - parameters: Dict[str, Any]

# TODO: Create DataMessage for content transmission
#   - content_type: str
#   - data: Any
#   - metadata: Dict[str, Any]

# TODO: Define QueryMessage for information requests
#   - query_type: str
#   - query: Any
#   - parameters: Dict[str, Any]

# TODO: Implement ResponseMessage for query replies
#   - query_id: UUID
#   - content: Any
#   - status: str
#   - errors: Optional[List[str]]

# TODO: Create EventMessage for system notifications
#   - event_type: str
#   - event_data: Any
#   - severity: str

# TODO: Add validation methods for each message type
# TODO: Implement helper methods for message creation
# TODO: Create serialization/deserialization methods