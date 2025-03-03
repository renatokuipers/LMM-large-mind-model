from typing import Dict, Any, Optional
import os
from pathlib import Path
from pydantic import BaseModel, Field, validator

class LLMSettings(BaseModel):
    """Configuration for the LLM."""
    
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


class AppSettings(BaseModel):
    """Application settings."""
    
    output_dir: str = Field("./generated", description="Output directory for generated projects")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")


class Config(BaseModel):
    """Application configuration."""
    
    llm: LLMSettings = Field(..., description="LLM settings")
    app: AppSettings = Field(..., description="Application settings")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            llm=LLMSettings(
                api_base_url=os.environ.get("LLM_API_BASE_URL", "http://192.168.2.12:1234"),
                model_name=os.environ.get("LLM_MODEL_NAME", "qwen2.5-7b-instruct"),
                api_key=os.environ.get("LLM_API_KEY"),
                max_context_length=int(os.environ.get("LLM_MAX_CONTEXT_LENGTH", "16384")),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "-1")),
                timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
            ),
            app=AppSettings(
                output_dir=os.environ.get("APP_OUTPUT_DIR", "./generated"),
                debug=os.environ.get("APP_DEBUG", "false").lower() == "true",
                log_level=os.environ.get("APP_LOG_LEVEL", "INFO")
            )
        )


# Default configuration
default_config = Config(
    llm=LLMSettings(
        api_base_url="http://localhost:5000/api",
        model_name="local-llm",
        api_key=None,
        max_context_length=8192,
        temperature=0.2,
        max_tokens=2048,
        timeout_seconds=120
    ),
    app=AppSettings(
        output_dir="./generated",
        debug=False,
        log_level="INFO"
    )
)


def get_config() -> Config:
    """Get application configuration.
    
    Returns:
        Application configuration
    """
    try:
        return Config.from_env()
    except Exception:
        return default_config