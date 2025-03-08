from typing import Any, Dict, List 
from pydantic import BaseModel, Field 
 
class MotherLLM(BaseModel): 
    """Interface to the 'Mother' LLM""" 
    llm_client: Any 
    tts_client: Any 
 
    class Config: 
        arbitrary_types_allowed = True 
