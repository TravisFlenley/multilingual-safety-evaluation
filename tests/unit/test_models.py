"""
Unit tests for model interfaces.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models import (
    BaseModel,
    ModelResponse,
    ModelRegistry,
    DummyModel,
    get_model_registry
)
import asyncio


class TestModelResponse:
    """Test ModelResponse class."""
    
    def test_model_response_creation(self):
        """Test creating model response."""
        response = ModelResponse(
            text="Test response",
            model="test-model",
            provider="test-provider",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"test": "data"},
            latency=1.5
        )
        
        assert response.text == "Test response"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.usage["prompt_tokens"] == 10
        assert response.latency == 1.5
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = ModelResponse(
            text="Test",
            model="model",
            provider="provider"
        )
        
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert response_dict["text"] == "Test"
        assert response_dict["model"] == "model"
        assert response_dict["provider"] == "provider"


class TestDummyModel:
    """Test DummyModel functionality."""
    
    @pytest.fixture
    def model(self):
        return DummyModel(model_name="test-dummy")
        
    def test_generate(self, model):
        """Test dummy model generation."""
        response = model.generate("Test prompt")
        
        assert isinstance(response, ModelResponse)
        assert response.model == "test-dummy"
        assert response.provider == "dummy"
        assert "dummy response" in response.text
        assert response.latency == 0.1
        
    @pytest.mark.asyncio
    async def test_generate_async(self, model):
        """Test async generation."""
        response = await model.generate_async("Test prompt")
        
        assert isinstance(response, ModelResponse)
        assert response.model == "test-dummy"
        
    def test_batch_generate(self, model):
        """Test batch generation."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = model.batch_generate(prompts)
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert isinstance(response, ModelResponse)
            assert f"Prompt {i+1}" in response.text
            
    def test_validate_response(self, model):
        """Test response validation."""
        valid_response = ModelResponse(
            text="Valid text",
            model="test",
            provider="test"
        )
        assert model.validate_response(valid_response)
        
        invalid_response = ModelResponse(
            text="",
            model="test",
            provider="test"
        )
        assert not model.validate_response(invalid_response)
        
        error_response = ModelResponse(
            text="Error",
            model="test",
            provider="test",
            metadata={"error": "Test error"}
        )
        assert not model.validate_response(error_response)
        
    def test_get_model_info(self, model):
        """Test getting model information."""
        info = model.get_model_info()
        
        assert info["provider"] == "dummy"
        assert info["model"] == "test-dummy"
        assert isinstance(info["config"], dict)


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    @pytest.fixture
    def registry(self):
        return ModelRegistry()
        
    def test_register_model(self, registry):
        """Test model registration."""
        model = DummyModel("test-model")
        registry.register_model("test", model)
        
        assert "test" in registry.list_models()
        assert registry.get_model("test") == model
        
    def test_register_default_model(self, registry):
        """Test setting default model."""
        model1 = DummyModel("model1")
        model2 = DummyModel("model2")
        
        registry.register_model("first", model1)
        assert registry._default_model == "first"
        
        registry.register_model("second", model2, set_default=True)
        assert registry._default_model == "second"
        
    def test_get_model_not_found(self, registry):
        """Test getting non-existent model."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            registry.get_model("nonexistent")
            
    def test_get_default_model(self, registry):
        """Test getting default model."""
        model = DummyModel("default")
        registry.register_model("default", model)
        
        # Should return default when no name specified
        assert registry.get_model() == model
        
    def test_list_models(self, registry):
        """Test listing registered models."""
        registry.register_model("model1", DummyModel("m1"))
        registry.register_model("model2", DummyModel("m2"))
        registry.register_model("model3", DummyModel("m3"))
        
        models = registry.list_models()
        assert len(models) == 3
        assert set(models) == {"model1", "model2", "model3"}
        
    def test_get_models_by_provider(self, registry):
        """Test getting models by provider."""
        registry.register_model("dummy1", DummyModel("d1"))
        registry.register_model("dummy2", DummyModel("d2"))
        
        dummy_models = registry.get_models_by_provider("dummy")
        assert len(dummy_models) == 2
        assert set(dummy_models) == {"dummy1", "dummy2"}
        
    def test_generate_with_all_models(self, registry):
        """Test generating with all models."""
        registry.register_model("model1", DummyModel("m1"))
        registry.register_model("model2", DummyModel("m2"))
        
        responses = registry.generate_with_all_models("Test prompt")
        
        assert len(responses) == 2
        assert "model1" in responses
        assert "model2" in responses
        
        for name, response in responses.items():
            assert isinstance(response, ModelResponse)
            assert response.text != ""
            
    @pytest.mark.asyncio
    async def test_generate_with_all_models_async(self, registry):
        """Test async generation with all models."""
        registry.register_model("model1", DummyModel("m1"))
        registry.register_model("model2", DummyModel("m2"))
        
        responses = await registry.generate_with_all_models_async("Test prompt")
        
        assert len(responses) == 2
        for name, response in responses.items():
            assert isinstance(response, ModelResponse)


class TestBaseModel:
    """Test BaseModel abstract class behavior."""
    
    def test_batch_generate_error_handling(self):
        """Test batch generation error handling."""
        class ErrorModel(BaseModel):
            def __init__(self):
                super().__init__("error-model")
                
            def generate(self, prompt, **kwargs):
                if "error" in prompt:
                    raise ValueError("Test error")
                return ModelResponse(
                    text="Success",
                    model=self.model_name,
                    provider=self.provider
                )
                
            async def generate_async(self, prompt, **kwargs):
                return self.generate(prompt, **kwargs)
                
        model = ErrorModel()
        prompts = ["Good prompt", "error prompt", "Another good prompt"]
        
        responses = model.batch_generate(prompts)
        
        assert len(responses) == 3
        assert responses[0].text == "Success"
        assert responses[1].text == ""  # Error case
        assert responses[1].metadata["error"] == "Test error"
        assert responses[2].text == "Success"
        
    @pytest.mark.asyncio
    async def test_batch_generate_async_with_semaphore(self):
        """Test async batch generation with concurrency limit."""
        
        call_times = []
        
        class TimedModel(BaseModel):
            def __init__(self):
                super().__init__("timed-model")
                
            def generate(self, prompt, **kwargs):
                return ModelResponse(
                    text=f"Response to {prompt}",
                    model=self.model_name,
                    provider=self.provider
                )
                
            async def generate_async(self, prompt, **kwargs):
                call_times.append(asyncio.get_event_loop().time())
                await asyncio.sleep(0.1)
                return self.generate(prompt, **kwargs)
                
        model = TimedModel()
        prompts = [f"Prompt {i}" for i in range(10)]
        
        # With max_concurrent=3, should see batching
        responses = await model.batch_generate_async(
            prompts, 
            max_concurrent=3
        )
        
        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert f"Prompt {i}" in response.text