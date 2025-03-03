from typing import Dict, List, Optional, Union, Any
import logging
import json
from pydantic import BaseModel, Field, validator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from schemas.llm_io import LLMRequest, LLMResponse, CodeGenerationRequest, CodeGenerationResponse
from core.code_memory import CodeMemory

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for the local LLM."""
    
    api_base_url: str = Field(..., description="Base URL for the LLM API")
    model_name: str = Field(..., description="Name of the model to use")
    api_key: Optional[str] = Field(None, description="API key if required")
    max_context_length: int = Field(8192, description="Maximum context length in tokens")
    temperature: float = Field(0.2, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    timeout_seconds: int = Field(120, description="Timeout for requests in seconds")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v


class LLMManager:
    """Manages interactions with the local LLM.
    
    This class handles prompting, context management, and response parsing
    for code generation tasks.
    """
    
    def __init__(self, config: LLMConfig, code_memory: Optional[CodeMemory] = None):
        """Initialize the LLM manager.
        
        Args:
            config: LLM configuration
            code_memory: Optional CodeMemory instance for context enhancement
        """
        self.config = config
        self.code_memory = code_memory
        self.client = httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def _build_system_prompt(self, task_type: str) -> str:
        """Build a system prompt for a specific task type.
        
        Args:
            task_type: Type of task (e.g., "code_generation", "architecture")
            
        Returns:
            System prompt string
        """
        # Different task types get different system prompts
        if task_type == "code_generation":
            return """You are an expert Python developer tasked with generating high-quality, 
            production-ready code. Follow these guidelines:
            1. Write clean, type-safe code with proper error handling
            2. Use Pydantic for data validation where appropriate
            3. Include detailed docstrings with examples
            4. Follow PEP 8 style conventions
            5. Implement exactly what is requested without placeholders
            6. Ensure the code is complete and functional
            7. Return structured output in the exact JSON format requested
            """
        elif task_type == "architecture":
            return """You are a software architect designing a high-quality system.
            Your task is to analyze requirements and produce a coherent architecture.
            Focus on:
            1. Clean separation of concerns
            2. Well-defined interfaces between components
            3. Appropriate design patterns
            4. Scalability and maintainability
            5. Return structured output in the exact JSON format requested
            """
        else:
            return """You are an AI assistant helping with software development.
            Provide detailed, accurate responses following the requested format.
            """
    
    def _enhance_context_with_code_memory(self, prompt: str) -> str:
        """Enhance the prompt with relevant code memory context.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt with code memory context
        """
        if not self.code_memory:
            return prompt
            
        # Extract code memory summary and append to prompt
        memory_summary = self.code_memory.get_summary_for_context(max_length=2000)
        
        enhanced_prompt = f"""
        # Project Code Context
        
        {memory_summary}
        
        # Task
        
        {prompt}
        """
        
        return enhanced_prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _send_request(self, request: LLMRequest) -> dict:
        """Send a request to the LLM API with retry logic.
        
        Args:
            request: LLM request
            
        Returns:
            Raw API response
            
        Raises:
            httpx.HTTPError: If the request fails after retries
        """
        payload = request.dict(exclude_none=True)
        
        response = await self.client.post(
            f"{self.config.api_base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
    async def generate(self, 
                      prompt: str, 
                      task_type: str = "general",
                      schema: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            task_type: Type of task for specialized system prompts
            schema: Optional response schema for structured output
            
        Returns:
            LLM response
            
        Raises:
            Exception: If the LLM request fails
        """
        # Enhance context if code memory is available
        enhanced_prompt = self._enhance_context_with_code_memory(prompt)
        
        # Build the request
        messages = [
            {"role": "system", "content": self._build_system_prompt(task_type)},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        request = LLMRequest(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"} if schema else None
        )
        
        try:
            response_data = await self._send_request(request)
            
            response = LLMResponse(
                content=response_data["choices"][0]["message"]["content"],
                usage=response_data.get("usage", {})
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating from LLM: {str(e)}")
            raise
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code based on a structured request.
        
        Args:
            request: Code generation request
            
        Returns:
            Structured code generation response
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        # Define the expected response schema
        response_schema = {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The generated code"},
                "explanation": {"type": "string", "description": "Explanation of the code"},
                "imports": {"type": "array", "items": {"type": "string"}, "description": "Required imports"},
                "dependencies": {"type": "array", "items": {"type": "string"}, "description": "Component dependencies"}
            },
            "required": ["code", "imports", "dependencies"]
        }
        
        # Build a detailed prompt for code generation
        prompt = f"""
        # Code Generation Task
        
        Generate the following code component:
        
        Component: {request.component_type} {request.name}
        Module: {request.module_path}
        
        ## Description
        {request.description}
        
        ## Requirements
        {request.requirements}
        
        {f'## Additional Context\n{request.additional_context}' if request.additional_context else ''}
        
        ## Expected Output Format
        Return a JSON object with these fields:
        - code: The complete implementation as a string
        - explanation: Brief explanation of the implementation
        - imports: Array of import statements needed
        - dependencies: Array of component names this depends on
        
        Do not include explanatory text outside the JSON.
        """
        
        try:
            # Generate with the structured schema
            response = await self.generate(
                prompt=prompt,
                task_type="code_generation",
                schema=response_schema
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse LLM response as JSON: {response.content}")
            
            # Validate and extract fields
            if not isinstance(response_data, dict) or "code" not in response_data:
                raise ValueError(f"Invalid response structure: {response_data}")
            
            code_response = CodeGenerationResponse(
                code=response_data["code"],
                explanation=response_data.get("explanation", ""),
                imports=response_data.get("imports", []),
                dependencies=response_data.get("dependencies", [])
            )
            
            return code_response
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise