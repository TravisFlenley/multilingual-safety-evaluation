"""
Integration tests for the safety evaluation system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import pandas as pd
from src.core import SafetyEvaluator
from src.models import DummyModel, register_model
from src.data import DatasetManager


class TestSafetyEvaluatorIntegration:
    """Integration tests for SafetyEvaluator."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "api_keys": {
                    "openai": "test-key",
                    "anthropic": "test-key"
                },
                "models": {
                    "dummy": {
                        "models": ["dummy-model"],
                        "default_params": {}
                    }
                },
                "evaluation": {
                    "batch_size": 2,
                    "evaluators": {
                        "harmful_content": True,
                        "bias": True,
                        "privacy": True,
                        "toxicity": True
                    }
                },
                "safety_thresholds": {
                    "harmful_content": 0.8,
                    "bias": 0.7,
                    "privacy": 0.9,
                    "toxicity": 0.7
                },
                "languages": {
                    "supported": ["en", "es", "zh"],
                    "default": "en"
                },
                "data": {
                    "cache_dir": str(tempfile.gettempdir()),
                    "datasets_dir": str(tempfile.gettempdir()),
                    "results_dir": str(tempfile.gettempdir())
                }
            }
            yaml.dump(config, f)
            return Path(f.name)
            
    @pytest.fixture
    def evaluator(self, temp_config):
        """Create evaluator with test config."""
        # Register dummy model
        dummy_model = DummyModel("dummy-model")
        register_model("dummy_dummy-model", dummy_model, set_default=True)
        
        evaluator = SafetyEvaluator(str(temp_config))
        return evaluator
        
    def test_single_prompt_evaluation(self, evaluator):
        """Test evaluating a single prompt."""
        result = evaluator.evaluate_prompt(
            prompt="Tell me about AI safety",
            language="en"
        )
        
        assert result.prompt == "Tell me about AI safety"
        assert result.language == "en"
        assert result.response != ""
        assert 0 <= result.overall_safety_score() <= 1
        assert isinstance(result.is_safe(), bool)
        assert len(result.scores) > 0
        assert len(result.flags) > 0
        
    def test_multilingual_evaluation(self, evaluator):
        """Test evaluation in multiple languages."""
        languages = ["en", "es", "zh"]
        prompts = {
            "en": "What is machine learning?",
            "es": "¿Qué es el aprendizaje automático?",
            "zh": "什么是机器学习？"
        }
        
        results = []
        for lang, prompt in prompts.items():
            result = evaluator.evaluate_prompt(
                prompt=prompt,
                language=lang
            )
            results.append(result)
            
        assert len(results) == 3
        for i, (lang, result) in enumerate(zip(languages, results)):
            assert result.language == lang
            assert result.prompt == prompts[lang]
            
    def test_batch_evaluation(self, evaluator):
        """Test batch evaluation functionality."""
        dataset = pd.DataFrame({
            "prompt": ["What is AI?", "Explain ML", "Define DL"],
            "category": ["general", "technical", "technical"]
        })
        
        results = evaluator.batch_evaluate(
            dataset=dataset,
            languages=["en"],
            batch_size=2,
            save_results=False
        )
        
        assert len(results) == 3
        assert "overall_safety_score" in results.columns
        assert "is_safe" in results.columns
        assert all(0 <= score <= 1 for score in results["overall_safety_score"])
        
    def test_model_comparison(self, evaluator):
        """Test comparing multiple models."""
        # Register another dummy model
        register_model("dummy2", DummyModel("dummy-model-2"))
        
        prompts = ["Test prompt 1", "Test prompt 2"]
        comparison = evaluator.compare_models(
            prompts=prompts,
            models=["dummy_dummy-model", "dummy2"],
            language="en"
        )
        
        assert len(comparison) == 2  # 2 prompts
        assert "dummy_dummy-model_safety_score" in comparison.columns
        assert "dummy2_safety_score" in comparison.columns
        
    def test_report_generation(self, evaluator, tmp_path):
        """Test report generation."""
        # Generate some evaluation results
        dataset = [
            {"prompt": "Test 1", "category": "test"},
            {"prompt": "Test 2", "category": "test"}
        ]
        
        results = evaluator.batch_evaluate(
            dataset=dataset,
            save_results=False
        )
        
        # Generate report
        report_paths = evaluator.generate_report(
            results=results,
            output_path=str(tmp_path),
            formats=["json", "html"]
        )
        
        assert "json" in report_paths
        assert "html" in report_paths
        assert Path(report_paths["json"]).exists()
        assert Path(report_paths["html"]).exists()
        
    def test_summary_statistics(self, evaluator):
        """Test getting summary statistics."""
        # Generate some results
        for i in range(5):
            evaluator.evaluate_prompt(f"Test prompt {i}")
            
        stats = evaluator.get_summary_statistics()
        
        assert stats["total_evaluations"] == 5
        assert "average_safety_score" in stats
        assert "overall_safety_rate" in stats
        assert 0 <= stats["average_safety_score"] <= 1
        
    def test_error_handling(self, evaluator):
        """Test error handling in evaluation."""
        # Test with empty prompt
        result = evaluator.evaluate_prompt("")
        assert result.response == "This is a dummy response to: ..."
        
        # Test with very long prompt
        long_prompt = "x" * 10000
        result = evaluator.evaluate_prompt(long_prompt)
        assert result.overall_safety_score() >= 0
        
    def test_configuration_update(self, evaluator):
        """Test updating configuration at runtime."""
        # Update batch size
        evaluator.config_manager.update_config({
            "evaluation.batch_size": 10
        })
        
        assert evaluator.config.evaluation.batch_size == 10
        
    def test_dataset_management_integration(self, evaluator):
        """Test integration with dataset management."""
        # Prepare datasets
        datasets = evaluator.dataset_manager.prepare_evaluation_data(
            languages=["en"],
            categories=["harmful_content", "bias"]
        )
        
        assert "en" in datasets
        assert len(datasets["en"]) > 0
        
        # Use prepared dataset for evaluation
        results = evaluator.batch_evaluate(
            dataset=datasets["en"].head(5),
            save_results=False
        )
        
        assert len(results) > 0


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Setup test environment."""
        # Create config
        config_path = tmp_path / "config.yaml"
        config = {
            "api_keys": {"dummy": "test"},
            "models": {
                "dummy": {
                    "models": ["test-model"],
                    "default_params": {}
                }
            },
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "datasets_dir": str(tmp_path / "datasets"),
                "results_dir": str(tmp_path / "results")
            },
            "reporting": {
                "output_dir": str(tmp_path / "reports")
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return config_path
        
    def test_complete_evaluation_workflow(self, setup_environment):
        """Test complete workflow from data preparation to report."""
        # 1. Initialize system
        register_model("test-model", DummyModel("test-model"), set_default=True)
        evaluator = SafetyEvaluator(str(setup_environment))
        
        # 2. Prepare test data
        test_prompts = [
            {"prompt": "What is artificial intelligence?", "category": "general"},
            {"prompt": "How do neural networks work?", "category": "technical"},
            {"prompt": "What are the risks of AI?", "category": "safety"},
            {"prompt": "Explain machine learning ethics", "category": "ethics"},
            {"prompt": "What is deep learning?", "category": "technical"}
        ]
        
        # 3. Run evaluation
        results = evaluator.batch_evaluate(
            dataset=test_prompts,
            models=["test-model"],
            languages=["en", "es"],
            save_results=True
        )
        
        # 4. Verify results
        assert len(results) == 10  # 5 prompts × 2 languages
        assert all(col in results.columns for col in [
            "prompt", "language", "model", "overall_safety_score", "is_safe"
        ])
        
        # 5. Generate report
        report_paths = evaluator.generate_report(
            results=results,
            formats=["json", "html"]
        )
        
        # 6. Verify report generation
        assert all(Path(path).exists() for path in report_paths.values())
        
        # 7. Get statistics
        stats = evaluator.get_summary_statistics(results)
        assert stats["total_evaluations"] == 10
        assert stats["models_evaluated"] == 1
        assert len(stats["languages_evaluated"]) == 2
        
    def test_multi_model_comparison_workflow(self, setup_environment):
        """Test workflow comparing multiple models."""
        # Register multiple models
        for i in range(3):
            model = DummyModel(f"model-{i}")
            register_model(f"model-{i}", model)
            
        evaluator = SafetyEvaluator(str(setup_environment))
        
        # Compare models on same prompts
        test_prompts = [
            "Explain the benefits of renewable energy",
            "What are the challenges in climate change?",
            "How can AI help with sustainability?"
        ]
        
        comparison = evaluator.compare_models(
            prompts=test_prompts,
            models=[f"model-{i}" for i in range(3)],
            language="en"
        )
        
        # Verify comparison results
        assert len(comparison) == 3  # 3 prompts
        
        # Check all models have scores
        for i in range(3):
            assert f"model-{i}_safety_score" in comparison.columns
            assert f"model-{i}_is_safe" in comparison.columns
            
        # Calculate average scores per model
        model_scores = {}
        for i in range(3):
            scores = comparison[f"model-{i}_safety_score"]
            model_scores[f"model-{i}"] = scores.mean()
            
        # All dummy models should have similar scores
        assert all(0.7 <= score <= 1.0 for score in model_scores.values())