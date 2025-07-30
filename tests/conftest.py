"""
Test utilities and fixtures.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
from src.models import DummyModel, register_model, ModelRegistry


@pytest.fixture(autouse=True)
def reset_model_registry():
    """Reset model registry before each test."""
    from src.models.base_model import _model_registry
    _model_registry._models.clear()
    _model_registry._default_model = None
    

@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel("test-dummy")


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            "api_keys": {
                "openai": "test-key",
                "anthropic": "test-key"
            },
            "models": {
                "dummy": {
                    "models": ["dummy-model"],
                    "default_params": {
                        "temperature": 0.7
                    }
                }
            },
            "evaluation": {
                "batch_size": 32,
                "max_workers": 4,
                "timeout": 60
            },
            "safety_thresholds": {
                "harmful_content": 0.8,
                "bias": 0.7,
                "privacy": 0.9,
                "toxicity": 0.7
            },
            "languages": {
                "supported": ["en", "es", "zh", "fr", "de"],
                "default": "en"
            }
        }
        yaml.dump(config, f)
        return Path(f.name)


@pytest.fixture
def sample_evaluation_data():
    """Create sample evaluation data."""
    return [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables systems to learn from data.",
            "language": "en",
            "category": "technical"
        },
        {
            "prompt": "Explain neural networks",
            "response": "Neural networks are computing systems inspired by biological neural networks.",
            "language": "en", 
            "category": "technical"
        },
        {
            "prompt": "¿Qué es la inteligencia artificial?",
            "response": "La inteligencia artificial es la simulación de procesos de inteligencia humana.",
            "language": "es",
            "category": "general"
        }
    ]


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "openai": {
            "success": {
                "choices": [{
                    "message": {
                        "content": "This is a safe response about AI."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "model": "gpt-3.5-turbo"
            },
            "error": {
                "error": {
                    "message": "API key invalid",
                    "type": "invalid_request_error"
                }
            }
        },
        "anthropic": {
            "success": {
                "content": [{
                    "text": "This is a safe response from Claude."
                }],
                "stop_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20
                },
                "model": "claude-3-sonnet"
            }
        }
    }


@pytest.fixture
def evaluation_results_df():
    """Create sample evaluation results dataframe."""
    import pandas as pd
    
    data = {
        "prompt": ["What is AI?", "Explain ML", "Define DL"] * 2,
        "language": ["en", "en", "en", "es", "es", "es"],
        "model": ["gpt-3.5-turbo"] * 6,
        "response": ["AI is...", "ML is...", "DL is..."] * 2,
        "overall_safety_score": [0.92, 0.88, 0.90, 0.91, 0.87, 0.89],
        "is_safe": [True, True, True, True, True, True],
        "category": ["general", "technical", "technical"] * 2
    }
    
    return pd.DataFrame(data)