"""
Unit tests for base evaluator functionality.
"""

import pytest
from src.evaluation import BaseEvaluator, CompositeEvaluator, EvaluationResult
from typing import Dict, Any


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""
    
    def __init__(self, name: str = "mock", score: float = 0.8):
        super().__init__()
        self.name = name
        self.default_score = score
        
    def evaluate(self, prompt: str, response: str, 
                language: str = "en", **kwargs) -> Dict[str, Any]:
        """Mock evaluation."""
        return {
            "scores": {f"{self.name}_score": self.default_score},
            "flags": {f"{self.name}_flag": self.default_score < 0.5},
            "metadata": {"evaluator": self.name}
        }


class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_evaluation_result_creation(self):
        """Test creating evaluation result."""
        result = EvaluationResult(
            prompt="test prompt",
            language="en",
            model="test-model",
            response="test response",
            scores={"safety": 0.9, "bias": 0.8},
            flags={"safe": True, "biased": False},
            metadata={"test": "data"},
            timestamp="2024-01-01T00:00:00"
        )
        
        assert result.prompt == "test prompt"
        assert result.language == "en"
        assert result.model == "test-model"
        assert result.response == "test response"
        assert result.scores["safety"] == 0.9
        assert result.flags["safe"] is True
        
    def test_overall_safety_score(self):
        """Test overall safety score calculation."""
        result = EvaluationResult(
            prompt="test",
            language="en",
            model="test",
            response="test",
            scores={
                "harmful_content": 0.9,
                "bias": 0.8,
                "privacy": 0.85,
                "misinformation": 0.7,
                "toxicity": 0.95
            },
            flags={},
            metadata={},
            timestamp="2024-01-01"
        )
        
        overall_score = result.overall_safety_score()
        assert 0 <= overall_score <= 1
        assert overall_score > 0.8  # Should be relatively high
        
    def test_is_safe(self):
        """Test safety determination."""
        # Safe result
        safe_result = EvaluationResult(
            prompt="test",
            language="en",
            model="test",
            response="test",
            scores={"harmful_content": 0.9, "bias": 0.85},
            flags={},
            metadata={},
            timestamp="2024-01-01"
        )
        assert safe_result.is_safe(threshold=0.8)
        
        # Unsafe result
        unsafe_result = EvaluationResult(
            prompt="test",
            language="en",
            model="test",
            response="test",
            scores={"harmful_content": 0.3, "bias": 0.4},
            flags={},
            metadata={},
            timestamp="2024-01-01"
        )
        assert not unsafe_result.is_safe(threshold=0.8)
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            prompt="test",
            language="en",
            model="test",
            response="test",
            scores={"safety": 0.9},
            flags={"safe": True},
            metadata={"meta": "data"},
            timestamp="2024-01-01"
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["prompt"] == "test"
        assert result_dict["scores"]["safety"] == 0.9
        assert result_dict["flags"]["safe"] is True


class TestBaseEvaluator:
    """Test BaseEvaluator functionality."""
    
    def test_batch_evaluate(self):
        """Test batch evaluation."""
        evaluator = MockEvaluator(score=0.9)
        
        data = [
            {"prompt": "prompt1", "response": "response1", "language": "en"},
            {"prompt": "prompt2", "response": "response2", "language": "es"},
            {"prompt": "prompt3", "response": "response3", "language": "zh"}
        ]
        
        results = evaluator.batch_evaluate(data)
        
        assert len(results) == 3
        for result in results:
            assert "scores" in result
            assert "mock_score" in result["scores"]
            assert result["scores"]["mock_score"] == 0.9
            
    def test_get_metrics(self):
        """Test getting evaluator metrics."""
        evaluator = MockEvaluator()
        metrics = evaluator.get_metrics()
        
        assert "evaluator" in metrics
        assert metrics["evaluator"] == "MockEvaluator"
        assert "supported_languages" in metrics
        assert "evaluation_dimensions" in metrics


class TestCompositeEvaluator:
    """Test CompositeEvaluator functionality."""
    
    def test_composite_evaluation(self):
        """Test evaluation with multiple evaluators."""
        evaluator1 = MockEvaluator(name="eval1", score=0.8)
        evaluator2 = MockEvaluator(name="eval2", score=0.6)
        evaluator3 = MockEvaluator(name="eval3", score=0.9)
        
        composite = CompositeEvaluator([evaluator1, evaluator2, evaluator3])
        
        result = composite.evaluate(
            prompt="test prompt",
            response="test response",
            language="en"
        )
        
        # Check all evaluators contributed
        assert "MockEvaluator_eval1_score" in result["scores"]
        assert "MockEvaluator_eval2_score" in result["scores"]
        assert "MockEvaluator_eval3_score" in result["scores"]
        
        # Check scores
        assert result["scores"]["MockEvaluator_eval1_score"] == 0.8
        assert result["scores"]["MockEvaluator_eval2_score"] == 0.6
        assert result["scores"]["MockEvaluator_eval3_score"] == 0.9
        
        # Check flags
        assert result["flags"]["MockEvaluator_eval2_flag"] is True  # Low score
        assert result["flags"]["MockEvaluator_eval1_flag"] is False
        
    def test_composite_error_handling(self):
        """Test composite evaluator handles errors gracefully."""
        
        class ErrorEvaluator(BaseEvaluator):
            def evaluate(self, prompt, response, language="en", **kwargs):
                raise ValueError("Test error")
                
        good_evaluator = MockEvaluator(name="good", score=0.9)
        bad_evaluator = ErrorEvaluator()
        
        composite = CompositeEvaluator([good_evaluator, bad_evaluator])
        
        result = composite.evaluate("test", "test")
        
        # Should still get results from good evaluator
        assert "MockEvaluator_good_score" in result["scores"]
        assert result["scores"]["MockEvaluator_good_score"] == 0.9
        
        # Should record error in metadata
        assert "ErrorEvaluator" in result["metadata"]
        assert "error" in result["metadata"]["ErrorEvaluator"]
        
    def test_get_evaluation_dimensions(self):
        """Test getting combined evaluation dimensions."""
        evaluator1 = MockEvaluator()
        evaluator1.config["dimensions"] = ["dim1", "dim2"]
        
        evaluator2 = MockEvaluator()
        evaluator2.config["dimensions"] = ["dim2", "dim3"]
        
        composite = CompositeEvaluator([evaluator1, evaluator2])
        dimensions = composite.get_evaluation_dimensions()
        
        assert set(dimensions) == {"dim1", "dim2", "dim3"}