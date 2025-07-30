"""
Base model interface for interacting with different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ModelResponse:
    """Container for model response data."""
    text: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    latency: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "metadata": self.metadata,
            "latency": self.latency
        }


class BaseModel(ABC):
    """Abstract base class for all model interfaces."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model interface.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.provider = self.__class__.__name__.replace("Model", "").lower()
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        pass
        
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate response asynchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        pass
        
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of ModelResponse objects
        """
        responses = []
        
        for prompt in prompts:
            try:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for prompt: {e}")
                responses.append(ModelResponse(
                    text="",
                    model=self.model_name,
                    provider=self.provider,
                    metadata={"error": str(e)}
                ))
                
        return responses
        
    async def batch_generate_async(self, prompts: List[str], 
                                  max_concurrent: int = 10, **kwargs) -> List[ModelResponse]:
        """
        Generate responses for multiple prompts asynchronously.
        
        Args:
            prompts: List of input prompts
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of ModelResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> ModelResponse:
            async with semaphore:
                try:
                    return await self.generate_async(prompt, **kwargs)
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    return ModelResponse(
                        text="",
                        model=self.model_name,
                        provider=self.provider,
                        metadata={"error": str(e)}
                    )
                    
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
        
    def validate_response(self, response: ModelResponse) -> bool:
        """
        Validate model response.
        
        Args:
            response: ModelResponse to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not response.text:
            return False
            
        if response.metadata and "error" in response.metadata:
            return False
            
        return True
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "config": self.config
        }
        
    def _measure_latency(self, func, *args, **kwargs):
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start_time
        return result, latency


class ModelRegistry:
    """Registry for managing multiple model interfaces."""
    
    def __init__(self):
        """Initialize model registry."""
        self._models = {}
        self._default_model = None
        
    def register_model(self, name: str, model: BaseModel, set_default: bool = False):
        """
        Register a model interface.
        
        Args:
            name: Unique name for the model
            model: Model interface instance
            set_default: Whether to set as default model
        """
        self._models[name] = model
        
        if set_default or self._default_model is None:
            self._default_model = name
            
        logger.info(f"Registered model: {name}")
        
    def get_model(self, name: Optional[str] = None) -> BaseModel:
        """
        Get model interface by name.
        
        Args:
            name: Model name (uses default if None)
            
        Returns:
            Model interface instance
        """
        model_name = name or self._default_model
        
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found in registry")
            
        return self._models[model_name]
        
    def list_models(self) -> List[str]:
        """Get list of registered model names."""
        return list(self._models.keys())
        
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get models for specific provider."""
        return [
            name for name, model in self._models.items()
            if model.provider == provider.lower()
        ]
        
    def generate_with_all_models(self, prompt: str, **kwargs) -> Dict[str, ModelResponse]:
        """
        Generate responses from all registered models.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping model names to responses
        """
        responses = {}
        
        for name, model in self._models.items():
            try:
                response = model.generate(prompt, **kwargs)
                responses[name] = response
            except Exception as e:
                logger.error(f"Error with model {name}: {e}")
                responses[name] = ModelResponse(
                    text="",
                    model=model.model_name,
                    provider=model.provider,
                    metadata={"error": str(e)}
                )
                
        return responses
        
    async def generate_with_all_models_async(self, prompt: str, **kwargs) -> Dict[str, ModelResponse]:
        """
        Generate responses from all models asynchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping model names to responses
        """
        tasks = {}
        
        for name, model in self._models.items():
            tasks[name] = model.generate_async(prompt, **kwargs)
            
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        responses = {}
        for (name, model), result in zip(self._models.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Error with model {name}: {result}")
                responses[name] = ModelResponse(
                    text="",
                    model=model.model_name,
                    provider=model.provider,
                    metadata={"error": str(result)}
                )
            else:
                responses[name] = result
                
        return responses


class DummyModel(BaseModel):
    """Dummy model for testing purposes."""
    
    def __init__(self, model_name: str = "dummy", config: Optional[Dict[str, Any]] = None):
        """Initialize dummy model."""
        super().__init__(model_name, config)
        
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate dummy response."""
        response_text = f"This is a dummy response to: {prompt[:50]}..."
        
        return ModelResponse(
            text=response_text,
            model=self.model_name,
            provider=self.provider,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split())},
            metadata={"dummy": True},
            latency=0.1
        )
        
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate dummy response asynchronously."""
        await asyncio.sleep(0.1)  # Simulate API latency
        return self.generate(prompt, **kwargs)


# Global model registry
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance."""
    return _model_registry


def register_model(name: str, model: BaseModel, set_default: bool = False):
    """Register a model in the global registry."""
    _model_registry.register_model(name, model, set_default)


def get_model(name: Optional[str] = None) -> BaseModel:
    """Get model from global registry."""
    return _model_registry.get_model(name)