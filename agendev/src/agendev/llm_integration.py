# llm_integration.py
"""Enhanced LLM client for diverse query types and context management."""

from typing import List, Dict, Union, Optional, Any, Tuple, Callable
import time
import json
import logging
from pathlib import Path
import threading
import random
from pydantic import BaseModel, Field

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
    max_retry_delay: float = 60.0  # Maximum retry delay in seconds
    retry_jitter: float = 0.1  # Jitter factor for randomizing retry delays
    
    # Circuit breaker configuration
    circuit_breaker_failures: int = 5  # Number of failures before circuit breaks
    circuit_breaker_reset_time: float = 300.0  # Time in seconds before circuit resets
    
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

class CircuitBreaker:
    """Circuit breaker for preventing repeated calls to a failing service."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 300.0):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before circuit breaks
            reset_timeout: Time in seconds before circuit resets
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False
    
    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        current_time = time.time()
        
        # Check if we should reset based on time since last failure
        if current_time - self.last_failure_time > self.reset_timeout:
            self.failure_count = 0
            self.is_open = False
        
        # Update failure count and time
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Check if circuit should open
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        self.failure_count = 0
        if self.is_open:
            self.is_open = False
            logger.info("Circuit breaker closed after successful operation")
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            True if request is allowed, False if circuit is open
        """
        current_time = time.time()
        
        # Check if we should try to reset a broken circuit
        if self.is_open and current_time - self.last_failure_time > self.reset_timeout:
            logger.info(f"Circuit breaker reset after {self.reset_timeout} seconds")
            self.is_open = False
            self.failure_count = 0
            return True
            
        return not self.is_open

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
        
        # Circuit breaker for handling service outages
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failures,
            reset_timeout=self.config.circuit_breaker_reset_time
        )
        
        # Track service health
        self.service_available = True
        self.last_health_check = 0
        self.health_check_interval = 60.0  # Seconds between health checks
    
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
    
    def _check_service_health(self, force: bool = False) -> bool:
        """
        Check if the LLM service is healthy.
        
        Args:
            force: Force a health check regardless of last check time
            
        Returns:
            True if service is healthy, False otherwise
        """
        current_time = time.time()
        
        # Only check health periodically unless forced
        if not force and current_time - self.last_health_check < self.health_check_interval:
            return self.service_available
            
        # Check service health
        try:
            self.last_health_check = current_time
            is_healthy = self.llm_client.health_check(timeout=5.0)
            
            # Update service status
            if is_healthy and not self.service_available:
                logger.info("LLM service is now available")
                self.service_available = True
                self.circuit_breaker.record_success()
            elif not is_healthy and self.service_available:
                logger.warning("LLM service is now unavailable")
                self.service_available = False
                self.circuit_breaker.record_failure()
                
            return is_healthy
        except Exception as e:
            logger.error(f"Error checking LLM service health: {e}")
            self.service_available = False
            self.circuit_breaker.record_failure()
            return False
    
    def _calculate_retry_delay(self, attempt: int, base_delay: float) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Current attempt number (1-based)
            base_delay: Base delay in seconds
            
        Returns:
            Delay in seconds for this attempt
        """
        # Calculate exponential backoff
        delay = min(
            self.config.max_retry_delay,
            base_delay * (2 ** (attempt - 1))
        )
        
        # Add jitter to avoid request storms
        jitter_factor = 1.0 - self.config.retry_jitter + (random.random() * self.config.retry_jitter * 2)
        return delay * jitter_factor
    
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
        
        # Check if circuit breaker allows request
        if not self.circuit_breaker.allow_request():
            logger.warning("Circuit breaker is open, request rejected")
            return self._generate_fallback_response(prompt, "service_unavailable")
        
        # Check service health
        if not self._check_service_health():
            logger.warning("LLM service is unavailable, using fallback response")
            return self._generate_fallback_response(prompt, "service_unavailable")
        
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
                completion_result = self._execute_with_retry(
                    lambda: self.llm_client.chat_completion(
                        messages=messages,
                        model=cfg.model,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens
                    ),
                    max_retries=cfg.max_retries,
                    base_delay=cfg.retry_delay_seconds
                )
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
            
            self.circuit_breaker.record_failure()
            
            # Return a fallback response
            return self._generate_fallback_response(prompt, "timeout")
        
        if completion_exception:
            logger.error(f"Error during LLM query: {completion_exception}")
            self.circuit_breaker.record_failure()
            
            # Add system message to context indicating error
            if save_to_context:
                self.context.add_message("system", f"Error: LLM query failed: {completion_exception}")
                
            return self._generate_fallback_response(prompt, "error")
        
        # Add assistant message to context if requested
        if save_to_context:
            self.context.add_message("assistant", completion_result)
        
        # Record success
        self.circuit_breaker.record_success()
        
        return completion_result
    
    def structured_query(
        self,
        prompt: str,
        json_schema: Dict,
        config: Optional[LLMConfig] = None,
        clear_context: bool = False,
        save_to_context: bool = True,
        timeout: Optional[float] = None
    ) -> Dict:
        """
        Send a structured query to the LLM and get a JSON response.
        
        Args:
            prompt: The prompt to send
            json_schema: Schema for the structured output
            config: Optional configuration override
            clear_context: Whether to clear context before sending
            save_to_context: Whether to save the interaction to context
            timeout: Request timeout in seconds
            
        Returns:
            Structured response from the LLM
        """
        # Check if circuit breaker allows request
        if not self.circuit_breaker.allow_request():
            logger.warning("Circuit breaker is open, structured query rejected")
            return self._generate_fallback_json_response(prompt, json_schema, "service_unavailable")
        
        # Check service health
        if not self._check_service_health():
            logger.warning("LLM service is unavailable, using fallback JSON response")
            return self._generate_fallback_json_response(prompt, json_schema, "service_unavailable")
        
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
        
        # Add schema information to the first system message or add a new one
        schema_hint = f"You must respond with JSON that matches this schema: {json.dumps(json_schema)}"
        if messages[0].role == "system":
            existing_content = messages[0].content
            if "JSON" not in existing_content and "json" not in existing_content:
                messages[0] = Message(role="system", content=f"{existing_content}\n\n{schema_hint}")
        else:
            messages.insert(0, Message(role="system", content=schema_hint))
        
        try:
            # Execute with retry logic and timeout handling
            response = self._execute_with_retry(
                lambda: self.llm_client.structured_completion(
                    messages=messages,
                    json_schema=json_schema,  # Still pass for the system message enhancement
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    stream=cfg.stream,
                    timeout=timeout
                ),
                max_retries=cfg.max_retries,
                base_delay=cfg.retry_delay_seconds,
                timeout=timeout
            )
            
            # Verify response is a dictionary
            if not isinstance(response, dict):
                logger.error(f"Unexpected response type: {type(response)}")
                # Try to parse it if it's a string
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse response as JSON: {response}")
                        response = self._generate_fallback_json_response(prompt, json_schema, "parse_error")
                else:
                    response = self._generate_fallback_json_response(prompt, json_schema, "type_error")
            
            # Save string representation of response to context if requested
            if save_to_context:
                response_str = json.dumps(response, indent=2) if isinstance(response, dict) else str(response)
                self.context.add_message("assistant", response_str)
            
            # Record success
            self.circuit_breaker.record_success()
            
            return response
            
        except TimeoutError as e:
            logger.error(f"Structured query timed out: {e}")
            self.circuit_breaker.record_failure()
            
            # Add error to context if requested
            if save_to_context:
                self.context.add_message("system", f"Error: Structured query timed out: {e}")
                
            return self._generate_fallback_json_response(prompt, json_schema, "timeout")
            
        except Exception as e:
            logger.error(f"Error during structured query: {e}")
            self.circuit_breaker.record_failure()
            
            # Add error to context if requested
            if save_to_context:
                self.context.add_message("system", f"Error: Structured query failed: {e}")
                
            return self._generate_fallback_json_response(prompt, json_schema, "error")
    
    def generate_embedding(self, content: str, model: Optional[str] = None) -> List[float]:
        """
        Generate an embedding for the given content.
        
        Args:
            content: Text to embed
            model: Optional embedding model override
            
        Returns:
            Embedding vector
        """
        # Check if circuit breaker allows request
        if not self.circuit_breaker.allow_request():
            logger.warning("Circuit breaker is open, embedding request rejected")
            return self._generate_fallback_embedding(content)
        
        # Check service health
        if not self._check_service_health():
            logger.warning("LLM service is unavailable, using fallback embedding")
            return self._generate_fallback_embedding(content)
        
        try:
            embedding = self._execute_with_retry(
                lambda: self.llm_client.get_embedding(
                    texts=content,
                    embedding_model=model or "text-embedding-nomic-embed-text-v1.5@q4_k_m"
                ),
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_delay_seconds
            )
            
            # Record success
            self.circuit_breaker.record_success()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            self.circuit_breaker.record_failure()
            return self._generate_fallback_embedding(content)
    
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
                # Handle language tag in first line
                if lines[0].startswith("```") and len(lines[0]) > 3:
                    lines = lines[1:-1]
                else:
                    lines = lines[1:-1]
                code = "\n".join(lines)
        
        # Save to file if path provided
        if file_path:
            resolved_path = resolve_path(file_path, create_parents=True)
            try:
                with open(resolved_path, 'w') as f:
                    f.write(code)
                logger.info(f"Successfully wrote generated code to {file_path}")
                
                # Create snapshot if requested
                if create_snapshot:
                    try:
                        save_snapshot(code, file_path, metadata={"language": language})
                        logger.info(f"Created snapshot for {file_path}")
                    except Exception as e:
                        logger.error(f"Error creating snapshot for {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error writing generated code to {file_path}: {e}")
        
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
    
    def _execute_with_retry(
        self, 
        operation: Callable, 
        max_retries: Optional[int] = None, 
        base_delay: Optional[float] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute an operation with retry logic and exponential backoff.
        
        Args:
            operation: Function to execute
            max_retries: Maximum number of retries
            base_delay: Base delay between retries in seconds
            timeout: Maximum time to wait for operation in seconds
            
        Returns:
            Result of the operation
            
        Raises:
            ValueError: For invalid arguments or API errors
            TimeoutError: If all retries fail with timeouts
            Exception: If all retries fail
        """
        retries = max_retries or self.config.max_retries
        delay = base_delay or self.config.retry_delay_seconds
        last_error = None
        
        for attempt in range(1, retries + 1):
            try:
                logger.debug(f"Attempt {attempt}/{retries} for operation")
                # Use the timeout if provided
                if timeout:
                    # Create a function that runs with timeout
                    def run_with_timeout():
                        return operation()
                    
                    result = None
                    exception = None
                    completed = False
                    
                    def worker():
                        nonlocal result, exception, completed
                        try:
                            result = operation()
                            completed = True
                        except Exception as e:
                            exception = e
                    
                    thread = threading.Thread(target=worker)
                    thread.daemon = True
                    thread.start()
                    
                    thread.join(timeout)
                    if not completed:
                        if thread.is_alive():
                            raise TimeoutError(f"Operation timed out after {timeout} seconds")
                        elif exception:
                            raise exception
                        else:
                            raise RuntimeError("Unknown error occurred")
                    
                    if exception:
                        raise exception
                        
                    return result
                else:
                    # Run without timeout
                    return operation()
                
            except Exception as e:
                last_error = e
                retry_delay = self._calculate_retry_delay(attempt, delay)
                
                # Don't wait after the last attempt
                if attempt < retries:
                    error_type = type(e).__name__
                    logger.warning(f"Attempt {attempt} failed with {error_type}: {e}. Retrying in {retry_delay:.2f}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {retries} attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        if isinstance(last_error, TimeoutError):
            raise TimeoutError(f"Operation timed out after {retries} attempts")
        
        raise last_error or RuntimeError("Unknown error during operation")
    
    def _generate_fallback_response(self, prompt: str, error_type: str) -> str:
        """
        Generate a fallback response when LLM service is unavailable.
        
        Args:
            prompt: The original prompt
            error_type: Type of error that occurred
            
        Returns:
            Fallback response text
        """
        if error_type == "service_unavailable":
            return (
                "I apologize, but I'm currently experiencing connectivity issues with the language model service. "
                "Please try again in a few minutes."
            )
        elif error_type == "timeout":
            return (
                "I apologize, but the request timed out. This might be due to high demand or complexity of the request. "
                "Please try again, possibly with a simpler query."
            )
        else:
            return (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your query."
            )
    
    def _generate_fallback_json_response(self, prompt: str, json_schema: Dict, error_type: str) -> Dict:
        """
        Generate a fallback JSON response when LLM service is unavailable.
        
        Args:
            prompt: The original prompt
            json_schema: The expected schema
            error_type: Type of error that occurred
            
        Returns:
            Fallback JSON response
        """
        # Create a minimal valid response based on the schema
        required_props = json_schema.get("schema", {}).get("required", [])
        properties = json_schema.get("schema", {}).get("properties", {})
        
        response = {}
        
        for prop in required_props:
            if prop in properties:
                prop_type = properties[prop].get("type", "string")
                
                if prop_type == "string":
                    response[prop] = self._get_fallback_message(error_type)
                elif prop_type == "number" or prop_type == "integer":
                    response[prop] = 0
                elif prop_type == "boolean":
                    response[prop] = False
                elif prop_type == "array":
                    response[prop] = []
                elif prop_type == "object":
                    response[prop] = {}
            else:
                # Default to string if property not found in schema
                response[prop] = self._get_fallback_message(error_type)
        
        # Add error information
        response["_error"] = {
            "type": error_type,
            "message": self._get_fallback_message(error_type)
        }
        
        return response
    
    def _get_fallback_message(self, error_type: str) -> str:
        """Get fallback message based on error type."""
        if error_type == "service_unavailable":
            return "Service temporarily unavailable. Please try again later."
        elif error_type == "timeout":
            return "Request timed out. Please try again."
        elif error_type == "parse_error":
            return "Error parsing response. Please try again."
        else:
            return "An error occurred. Please try again."
    
    def _generate_fallback_embedding(self, content: str) -> List[float]:
        """
        Generate a fallback embedding when the service is unavailable.
        
        Args:
            content: Original content to embed
            
        Returns:
            Placeholder embedding vector
        """
        # Create a deterministic but unique embedding based on content hash
        import hashlib
        
        # Generate a hash of the content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Convert hash to a list of floats (normalized to -1 to 1)
        # This is not a real embedding but will be consistent for the same input
        hash_bytes = bytes.fromhex(content_hash)
        dimension = 384  # Common embedding dimension
        
        # Ensure we have enough bytes by repeating the hash if necessary
        while len(hash_bytes) < dimension:
            hash_bytes += bytes.fromhex(hashlib.md5(hash_bytes).hexdigest())
        
        # Convert bytes to floats between -1 and 1
        embedding = []
        for i in range(dimension):
            byte_val = hash_bytes[i]
            float_val = (byte_val / 127.5) - 1.0
            embedding.append(float_val)
        
        # Log warning
        logger.warning(f"Generated fallback embedding for content (length {len(content)}) due to service unavailability")
        
        return embedding