"""
OpenAI model interface implementation.
"""

import openai
from typing import Dict, Any, Optional
import asyncio
from loguru import logger
from .base_model import BaseModel, ModelResponse
import time


class OpenAIModel(BaseModel):
    """Interface for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", 
                 api_key: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI model interface.
        
        Args:
            model_name: Name of the OpenAI model
            api_key: OpenAI API key
            config: Additional configuration
        """
        super().__init__(model_name, config)
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif "api_key" in self.config:
            openai.api_key = self.config["api_key"]
            
        # Default parameters
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Update with config
        if "default_params" in self.config:
            self.default_params.update(self.config["default_params"])
            
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response from OpenAI model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        try:
            start_time = time.time()
            
            # Check if using chat model
            if self.model_name.startswith("gpt-"):
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                
                text = response.choices[0].message.content
                usage = response.usage.to_dict() if hasattr(response.usage, 'to_dict') else dict(response.usage)
                
            else:
                # Legacy completion API
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    **params
                )
                
                text = response.choices[0].text
                usage = response.usage.to_dict() if hasattr(response.usage, 'to_dict') else dict(response.usage)
                
            latency = time.time() - start_time
            
            return ModelResponse(
                text=text.strip(),
                model=self.model_name,
                provider="openai",
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model_version": response.model
                },
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
        # OpenAI doesn't have native async support yet, so we use asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, **kwargs)
        
    def validate_api_key(self) -> bool:
        """Validate if API key is set and working."""
        try:
            # Make a minimal API call to check if key is valid
            openai.Model.list()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
            
    def list_available_models(self) -> list:
        """List available OpenAI models."""
        try:
            models = openai.Model.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []