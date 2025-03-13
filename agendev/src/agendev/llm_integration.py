# llm_integration.py
"""Enhanced LLM client for diverse query types and context management."""

from typing import List, Dict, Union, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
import time
import json
import logging
from pathlib import Path
import threading

from .llm_module import LLMClient, Message
from .models.task_models import Task, TaskType
from .utils.fs_utils import save_json, load_json, resolve_path, save_snapshot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for LLM requests."""
    model: str = "qwen2.5-7b-instruct"
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(2000, ge=0)
    stream: bool = False
    context_window_size: int = Field(16384, ge=0)
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    
    # Context management
    include_history: bool = True
    max_history_messages: int = 10

class ContextWindow(BaseModel):
    """Manages conversation history and context."""
    system_message: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    max_messages: int = 20
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the context window."""
        self.messages.append({"role": role, "content": content})
        # Trim history if necessary
        if len(self.messages) > self.max_messages:
            # Keep system message if present, otherwise trim oldest messages
            if self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
            else:
                self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the context window."""
        return self.messages.copy()
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear all messages except optionally the system message."""
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def as_message_list(self) -> List[Message]:
        """
        Convert to a list of Message objects for LLMClient.
        
        Returns:
            List of Message objects, never empty
        """
        # Create message objects from internal messages
        message_list = [Message(role=msg["role"], content=msg["content"]) for msg in self.messages]
        
        # If messages list is empty, add a default system message
        if not message_list:
            default_message = Message(
                role="system", 
                content="You are a helpful AI assistant. Please provide a useful response."
            )
            message_list.append(default_message)
            
        return message_list

class LLMIntegration:
    """Enhanced LLM client for diverse query types and context management."""
    
    def __init__(
        self,
        base_url: str = "http://192.168.2.12:1234",
        config: Optional[LLMConfig] = None,
        system_message: Optional[str] = None
    ):
        """
        Initialize the LLM integration layer.
        
        Args:
            base_url: URL for the LLM API
            config: Configuration for LLM requests
            system_message: Default system message for all conversations
        """
        self.llm_client = LLMClient(base_url=base_url)
        self.config = config or LLMConfig()
        self.context = ContextWindow(max_messages=self.config.max_history_messages)
        
        # Set system message if provided
        if system_message:
            self.set_system_message(system_message)
            
        # Store task-specific configurations
        self.task_configs: Dict[TaskType, LLMConfig] = {}
        
        # Initialize with some defaults for different task types
        self._initialize_task_configs()
    
    def _initialize_task_configs(self) -> None:
        """Initialize default configurations for different task types."""
        self.task_configs = {
            TaskType.IMPLEMENTATION: LLMConfig(temperature=0.7, max_tokens=2000),
            TaskType.REFACTOR: LLMConfig(temperature=0.5, max_tokens=2000),
            TaskType.BUGFIX: LLMConfig(temperature=0.2, max_tokens=1500),
            TaskType.TEST: LLMConfig(temperature=0.4, max_tokens=2000),
            TaskType.DOCUMENTATION: LLMConfig(temperature=0.8, max_tokens=1500),
            TaskType.PLANNING: LLMConfig(temperature=0.9, max_tokens=3000),
        }
    
    def set_system_message(self, message: str) -> None:
        """Set or update the system message."""
        if self.context.messages and self.context.messages[0]["role"] == "system":
            self.context.messages[0]["content"] = message
        else:
            self.context.messages.insert(0, {"role": "system", "content": message})
    
    def query(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        clear_context: bool = False,
        save_to_context: bool = True,
        timeout: float = 30.0
    ) -> str:
        """
        Send a query to the LLM and return the response.
        
        Args:
            prompt: The prompt to send to the LLM
            config: LLM configuration options
            clear_context: Whether to clear the context before adding the prompt
            save_to_context: Whether to save the prompt and response to the context
            timeout: Maximum time in seconds to wait for LLM response
            
        Returns:
            Response from the LLM
        """
        cfg = config or self.config
        
        # Optionally clear context
        if clear_context:
            self.context.clear()
        
        # Add user message to context if requested
        if save_to_context:
            self.context.add_message("user", prompt)
        
        # Prepare message list
        messages = self.context.as_message_list() if cfg.include_history else [Message(role="user", content=prompt)]
        
        # Add timeout handling
        completion_result = None
        completion_exception = None
        completion_completed = False
        
        def run_completion_with_timeout():
            nonlocal completion_result, completion_exception, completion_completed
            try:
                # Execute chat completion with retry
                for attempt in range(1, cfg.max_retries + 1):
                    try:
                        completion_result = self.llm_client.chat_completion(
                            messages=messages,
                            model=cfg.model,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens
                        )
                        break
                    except Exception as e:
                        if attempt < cfg.max_retries:
                            backoff = 2 ** (attempt - 1)
                            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {backoff}s...")
                            time.sleep(backoff)
                        else:
                            logger.error(f"All {cfg.max_retries} attempts failed.")
                            raise
                
                completion_completed = True
            except Exception as e:
                completion_exception = e
        
        # Start completion in a separate thread
        completion_thread = threading.Thread(target=run_completion_with_timeout)
        completion_thread.daemon = True
        completion_thread.start()
        
        # Wait for completion with timeout
        start_time = time.time()
        while not completion_completed and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not completion_completed:
            error_msg = f"LLM query timed out after {timeout} seconds"
            logger.error(error_msg)
            if completion_exception:
                logger.error(f"Exception in LLM thread: {completion_exception}")
            
            # Add system message to context indicating timeout
            if save_to_context:
                self.context.add_message("system", f"Error: LLM query timed out after {timeout} seconds")
            
            # Return a fallback response
            return f"I apologize, but I'm having trouble generating a response within the time limit. Please try again or simplify your request."
        
        if completion_exception:
            raise completion_exception
        
        # Add assistant message to context if requested
        if save_to_context:
            self.context.add_message("assistant", completion_result)
        
        return completion_result
    
    def structured_query(
        self,
        prompt: str,
        json_schema: Dict,
        config: Optional[LLMConfig] = None,
        clear_context: bool = False,
        save_to_context: bool = True
    ) -> Dict:
        """
        Send a structured query to the LLM and get a JSON response.
        
        Args:
            prompt: The prompt to send
            json_schema: Schema for the structured output
            config: Optional configuration override
            clear_context: Whether to clear context before sending
            save_to_context: Whether to save the interaction to context
            
        Returns:
            Structured response from the LLM
        """
        # Use provided config or default
        cfg = config or self.config
        
        # Optionally clear context
        if clear_context:
            self.context.clear()
        
        # Add user message to context if requested
        if save_to_context:
            self.context.add_message("user", prompt)
        
        # Prepare message list
        messages = self.context.as_message_list() if cfg.include_history else [Message(role="user", content=prompt)]
        
        # Add system message if not present
        if not messages or messages[0].role != "system":
            if self.context.system_message:
                messages.insert(0, Message(role="system", content=self.context.system_message))
        
        # FIX: Ensure messages are not empty
        if not messages:
            # If somehow there are no messages, add the prompt as a user message
            messages = [Message(role="user", content=prompt)]
            logger.warning("No messages in context, adding prompt as user message")
        
        # Execute with retry logic
        response = self._execute_with_retry(
            lambda: self.llm_client.structured_completion(
                messages=messages,
                json_schema=json_schema,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=cfg.stream
            )
        )
        
        # Save string representation of response to context if requested
        if save_to_context:
            response_str = json.dumps(response, indent=2) if isinstance(response, dict) else str(response)
            self.context.add_message("assistant", response_str)
        
        return response
    
    def generate_embedding(self, content: str, model: Optional[str] = None) -> List[float]:
        """
        Generate an embedding for the given content.
        
        Args:
            content: Text to embed
            model: Optional embedding model override
            
        Returns:
            Embedding vector
        """
        return self._execute_with_retry(
            lambda: self.llm_client.get_embedding(
                texts=content,
                embedding_model=model or "text-embedding-nomic-embed-text-v1.5@q4_k_m"
            )
        )
    
    def clear_context(self, keep_system: bool = True) -> None:
        """
        Clear the context window.
        
        Args:
            keep_system: Whether to keep the system message
        """
        self.context.clear(keep_system=keep_system)
    
    def generate_code(
        self,
        prompt: str,
        language: str,
        file_path: Optional[str] = None,
        include_comments: bool = True,
        config: Optional[LLMConfig] = None,
        create_snapshot: bool = True
    ) -> str:
        """
        Generate code with an enhanced prompt for better structure preservation.
        
        Args:
            prompt: Description of the code to generate
            language: Programming language for the code
            file_path: Optional path to save the code
            include_comments: Whether to include comments in the generated code
            config: Optional configuration override
            create_snapshot: Whether to create a snapshot of the generated code
            
        Returns:
            Generated code
        """
        # Use lower temperature for code generation
        code_config = config or LLMConfig(temperature=0.4, max_tokens=3000)
        
        # Set system message for code generation
        system_message = f"""You are an expert {language} developer. 
        Generate clean, well-structured {language} code that follows best practices.
        {'Include detailed comments explaining the code.' if include_comments else 'Minimize comments, focus on clean code.'}
        Output ONLY the code with no additional text or explanations outside of code comments."""
        
        # Build the structured prompt
        structured_prompt = f"""
        Task: Generate {language} code
        
        Requirements:
        {prompt}
        
        Please generate the complete code. Do not include markdown code blocks or any explanatory text outside the code.
        """
        
        # Generate the code
        code = self.query(
            prompt=structured_prompt,
            config=code_config,
            clear_context=True,
            save_to_context=False
        )
        
        # Strip any markdown code blocks if present
        code = code.strip()
        if code.startswith("```") and code.endswith("```"):
            lines = code.split("\n")
            if len(lines) >= 2:
                code = "\n".join(lines[1:-1])
        
        # Save to file if path provided
        if file_path:
            resolved_path = resolve_path(file_path, create_parents=True)
            with open(resolved_path, 'w') as f:
                f.write(code)
            
            # Create snapshot if requested
            if create_snapshot:
                save_snapshot(code, file_path, metadata={"language": language})
        
        return code
    
    def get_task_config(self, task: Task) -> LLMConfig:
        """
        Get an appropriate configuration for a specific task.
        
        Args:
            task: The task to get configuration for
            
        Returns:
            LLM configuration tailored to the task
        """
        # Start with the default for this task type
        base_config = self.task_configs.get(task.task_type, self.config)
        
        # Override with task-specific parameters if available
        return LLMConfig(
            model=base_config.model,
            temperature=task.temperature if task.temperature is not None else base_config.temperature,
            max_tokens=task.max_tokens if task.max_tokens is not None else base_config.max_tokens,
            stream=base_config.stream,
            context_window_size=base_config.context_window_size,
            max_retries=base_config.max_retries,
            retry_delay_seconds=base_config.retry_delay_seconds,
            include_history=base_config.include_history,
            max_history_messages=base_config.max_history_messages
        )
    
    def update_task_config(self, task_type: TaskType, config: LLMConfig) -> None:
        """
        Update the configuration for a specific task type.
        
        Args:
            task_type: The task type to update configuration for
            config: The new configuration
        """
        self.task_configs[task_type] = config
    
    def _execute_with_retry(self, operation: Callable, max_retries: Optional[int] = None) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Function to execute
            max_retries: Maximum number of retries
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.config.retry_delay_seconds}s...")
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(f"All {retries} attempts failed.")
        
        raise last_error