"""
Anthropic model interface implementation.
"""

import anthropic
from typing import Dict, Any, Optional
import asyncio
from loguru import logger
from .base_model import BaseModel, ModelResponse
import time


class AnthropicModel(BaseModel):
    """Interface for Anthropic models."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229",
                 api_key: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Anthropic model interface.
        
        Args:
            model_name: Name of the Anthropic model
            api_key: Anthropic API key
            config: Additional configuration
        """
        super().__init__(model_name, config)
        
        # Initialize client
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif "api_key" in self.config:
            self.client = anthropic.Anthropic(api_key=self.config["api_key"])
        else:
            self.client = anthropic.Anthropic()  # Will use ANTHROPIC_API_KEY env var
            
        # Default parameters
        self.default_params = {
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "stop_sequences": None
        }
        
        # Update with config
        if "default_params" in self.config:
            self.default_params.update(self.config["default_params"])
            
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response from Anthropic model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            start_time = time.time()
            
            # Create message
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **params
            )
            
            latency = time.time() - start_time
            
            # Extract text from response
            text = ""
            if response.content:
                for content in response.content:
                    if hasattr(content, 'text'):
                        text += content.text
                        
            return ModelResponse(
                text=text.strip(),
                model=self.model_name,
                provider="anthropic",
                usage={
                    "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else None,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else None
                },
                metadata={
                    "stop_reason": response.stop_reason if hasattr(response, 'stop_reason') else None,
                    "model_version": response.model
                },
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
            
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response asynchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        # Use asyncio to run synchronous method
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, **kwargs)
        
    def validate_api_key(self) -> bool:
        """Validate if API key is set and working."""
        try:
            # Make a minimal API call to check if key is valid
            self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
            
    def get_supported_models(self) -> list:
        """Get list of supported Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]