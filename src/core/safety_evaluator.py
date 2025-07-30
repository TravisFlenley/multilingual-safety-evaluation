"""
Core safety evaluator that orchestrates the entire evaluation process.
"""

import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..evaluation import (
    CompositeEvaluator,
    EvaluationResult,
    HarmfulContentEvaluator,
    BiasEvaluator,
    PrivacyEvaluator,
    ToxicityEvaluator
)
from ..models import ModelRegistry, get_model_registry, ModelResponse
from ..data import DatasetManager
from ..utils import ConfigManager, get_config_manager
from .report_generator import ReportGenerator


class SafetyEvaluator:
    """Main class for conducting safety evaluations on LLMs."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize safety evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.model_registry = get_model_registry()
        self.dataset_manager = DatasetManager()
        self.report_generator = ReportGenerator(self.config)
        
        # Initialize evaluators
        self._initialize_evaluators()
        
        # Setup models
        self._setup_models()
        
        # Results storage
        self.results = []
        
    def _initialize_evaluators(self):
        """Initialize safety evaluators based on configuration."""
        evaluators = []
        
        # Add evaluators based on configuration
        if self.config.get("evaluation.evaluators.harmful_content", True):
            evaluators.append(HarmfulContentEvaluator(
                config=self.config.get("safety_thresholds.harmful_content", {})
            ))
            
        if self.config.get("evaluation.evaluators.bias", True):
            evaluators.append(BiasEvaluator(
                config=self.config.get("safety_thresholds.bias", {})
            ))
            
        if self.config.get("evaluation.evaluators.privacy", True):
            evaluators.append(PrivacyEvaluator(
                config=self.config.get("safety_thresholds.privacy", {})
            ))
            
        if self.config.get("evaluation.evaluators.toxicity", True):
            evaluators.append(ToxicityEvaluator(
                config=self.config.get("safety_thresholds.toxicity", {})
            ))
            
        # Create composite evaluator
        self.evaluator = CompositeEvaluator(evaluators)
        
        logger.info(f"Initialized {len(evaluators)} safety evaluators")
        
    def _setup_models(self):
        """Setup model interfaces based on configuration."""
        # Import model classes
        from ..models import OpenAIModel, AnthropicModel
        
        # Setup OpenAI models
        openai_key = self.config_manager.get_api_key("openai")
        if openai_key:
            for model in self.config.get("models.openai.models", []):
                try:
                    openai_model = OpenAIModel(
                        model_name=model,
                        api_key=openai_key,
                        config=self.config.get("models.openai.default_params", {})
                    )
                    self.model_registry.register_model(f"openai_{model}", openai_model)
                    logger.info(f"Registered OpenAI model: {model}")
                except Exception as e:
                    logger.error(f"Failed to setup OpenAI model {model}: {e}")
                    
        # Setup Anthropic models
        anthropic_key = self.config_manager.get_api_key("anthropic")
        if anthropic_key:
            for model in self.config.get("models.anthropic.models", []):
                try:
                    anthropic_model = AnthropicModel(
                        model_name=model,
                        api_key=anthropic_key,
                        config=self.config.get("models.anthropic.default_params", {})
                    )
                    self.model_registry.register_model(f"anthropic_{model}", anthropic_model)
                    logger.info(f"Registered Anthropic model: {model}")
                except Exception as e:
                    logger.error(f"Failed to setup Anthropic model {model}: {e}")
                    
    def evaluate_prompt(self, prompt: str, language: str = "en", 
                       model: Optional[str] = None, **kwargs) -> EvaluationResult:
        """
        Evaluate a single prompt with specified model.
        
        Args:
            prompt: Input prompt to evaluate
            language: Language code
            model: Model name (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult object
        """
        # Get model
        model_interface = self.model_registry.get_model(model)
        
        # Generate response
        try:
            response = model_interface.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            response = ModelResponse(
                text="",
                model=model_interface.model_name,
                provider=model_interface.provider,
                metadata={"error": str(e)}
            )
            
        # Evaluate response
        eval_result = self.evaluator.evaluate(
            prompt=prompt,
            response=response.text,
            language=language
        )
        
        # Create evaluation result
        result = EvaluationResult(
            prompt=prompt,
            language=language,
            model=model or model_interface.model_name,
            response=response.text,
            scores=eval_result.get("scores", {}),
            flags=eval_result.get("flags", {}),
            metadata={
                **eval_result.get("metadata", {}),
                "model_metadata": response.metadata,
                "latency": response.latency
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Store result
        self.results.append(result)
        
        return result
        
    def batch_evaluate(self, dataset: Union[pd.DataFrame, str, List[Dict[str, str]]],
                      models: Optional[List[str]] = None,
                      languages: Optional[List[str]] = None,
                      batch_size: Optional[int] = None,
                      save_results: bool = True) -> pd.DataFrame:
        """
        Evaluate multiple prompts across models and languages.
        
        Args:
            dataset: DataFrame, file path, or list of prompts
            models: List of model names to evaluate
            languages: List of language codes
            batch_size: Batch size for processing
            save_results: Whether to save results to disk
            
        Returns:
            DataFrame with evaluation results
        """
        # Load dataset
        if isinstance(dataset, str):
            dataset = self.dataset_manager.collector.load_dataset(dataset)
        elif isinstance(dataset, list):
            dataset = pd.DataFrame(dataset)
            
        # Get models and languages
        models = models or self.model_registry.list_models()
        languages = languages or self.config_manager.get_supported_languages()
        batch_size = batch_size or self.config.get("evaluation.batch_size", 32)
        
        logger.info(f"Starting batch evaluation: {len(dataset)} prompts, "
                   f"{len(models)} models, {len(languages)} languages")
        
        results = []
        
        # Progress tracking
        total_evaluations = len(dataset) * len(models) * len(languages)
        pbar = tqdm(total=total_evaluations, desc="Evaluating")
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset.iloc[i:i + batch_size]
            
            for _, row in batch.iterrows():
                prompt = row.get("prompt", "")
                
                for language in languages:
                    for model in models:
                        try:
                            result = self.evaluate_prompt(
                                prompt=prompt,
                                language=language,
                                model=model
                            )
                            
                            # Convert to dict for DataFrame
                            result_dict = result.to_dict()
                            result_dict.update({
                                "dataset_id": row.get("id", ""),
                                "category": row.get("category", ""),
                                "overall_safety_score": result.overall_safety_score(),
                                "is_safe": result.is_safe()
                            })
                            
                            results.append(result_dict)
                            
                        except Exception as e:
                            logger.error(f"Evaluation error: {e}")
                            results.append({
                                "prompt": prompt,
                                "language": language,
                                "model": model,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat()
                            })
                            
                        pbar.update(1)
                        
        pbar.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if requested
        if save_results:
            output_path = self._save_results(results_df)
            logger.info(f"Results saved to: {output_path}")
            
        return results_df
        
    async def batch_evaluate_async(self, dataset: Union[pd.DataFrame, str, List[Dict[str, str]]],
                                  models: Optional[List[str]] = None,
                                  languages: Optional[List[str]] = None,
                                  max_concurrent: int = 10) -> pd.DataFrame:
        """
        Evaluate multiple prompts asynchronously.
        
        Args:
            dataset: DataFrame, file path, or list of prompts
            models: List of model names to evaluate
            languages: List of language codes
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            DataFrame with evaluation results
        """
        # Load dataset
        if isinstance(dataset, str):
            dataset = self.dataset_manager.collector.load_dataset(dataset)
        elif isinstance(dataset, list):
            dataset = pd.DataFrame(dataset)
            
        # Get models and languages
        models = models or self.model_registry.list_models()
        languages = languages or self.config_manager.get_supported_languages()
        
        # Create evaluation tasks
        tasks = []
        for _, row in dataset.iterrows():
            prompt = row.get("prompt", "")
            
            for language in languages:
                for model in models:
                    task = self._evaluate_prompt_async(prompt, language, model)
                    tasks.append(task)
                    
        # Run tasks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task
                
        results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
        
        # Convert to DataFrame
        return pd.DataFrame([r.to_dict() for r in results if r])
        
    async def _evaluate_prompt_async(self, prompt: str, language: str, model: str) -> Optional[EvaluationResult]:
        """Evaluate prompt asynchronously."""
        try:
            # This would be properly async in production
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.evaluate_prompt, prompt, language, model)
        except Exception as e:
            logger.error(f"Async evaluation error: {e}")
            return None
            
    def generate_report(self, results: Optional[pd.DataFrame] = None,
                       output_path: Optional[str] = None,
                       formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate evaluation report.
        
        Args:
            results: Results DataFrame (uses stored results if None)
            output_path: Output directory path
            formats: List of output formats (html, pdf, json)
            
        Returns:
            Dictionary mapping format to file path
        """
        # Use stored results if not provided
        if results is None:
            if not self.results:
                raise ValueError("No results available to generate report")
            results = pd.DataFrame([r.to_dict() for r in self.results])
            
        # Generate report
        return self.report_generator.generate(
            results=results,
            output_path=output_path,
            formats=formats
        )
        
    def compare_models(self, prompts: List[str], 
                      models: Optional[List[str]] = None,
                      language: str = "en") -> pd.DataFrame:
        """
        Compare multiple models on the same prompts.
        
        Args:
            prompts: List of prompts to evaluate
            models: List of model names (uses all if None)
            language: Language code
            
        Returns:
            DataFrame with comparison results
        """
        models = models or self.model_registry.list_models()
        
        comparison_results = []
        
        for prompt in prompts:
            prompt_results = {"prompt": prompt, "language": language}
            
            for model in models:
                result = self.evaluate_prompt(prompt, language, model)
                
                # Add model results
                prompt_results[f"{model}_response"] = result.response
                prompt_results[f"{model}_safety_score"] = result.overall_safety_score()
                prompt_results[f"{model}_is_safe"] = result.is_safe()
                
                # Add individual scores
                for score_name, score_value in result.scores.items():
                    prompt_results[f"{model}_{score_name}"] = score_value
                    
            comparison_results.append(prompt_results)
            
        return pd.DataFrame(comparison_results)
        
    def _save_results(self, results: pd.DataFrame) -> str:
        """Save evaluation results to disk."""
        output_dir = Path(self.config.get("data.results_dir", "data/results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.parquet"
        filepath = output_dir / filename
        
        results.to_parquet(filepath, index=False)
        
        # Also save as JSON for readability
        json_filepath = filepath.with_suffix(".json")
        results.to_json(json_filepath, orient="records", indent=2)
        
        return str(filepath)
        
    def get_summary_statistics(self, results: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get summary statistics from evaluation results.
        
        Args:
            results: Results DataFrame (uses stored results if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if results is None:
            if not self.results:
                return {}
            results = pd.DataFrame([r.to_dict() for r in self.results])
            
        summary = {
            "total_evaluations": len(results),
            "models_evaluated": results["model"].nunique() if "model" in results else 0,
            "languages_evaluated": results["language"].unique().tolist() if "language" in results else [],
            "overall_safety_rate": results["is_safe"].mean() if "is_safe" in results else 0,
            "average_safety_score": results["overall_safety_score"].mean() if "overall_safety_score" in results else 0
        }
        
        # Per-model statistics
        if "model" in results and "overall_safety_score" in results:
            summary["per_model_stats"] = results.groupby("model").agg({
                "overall_safety_score": ["mean", "std", "min", "max"],
                "is_safe": "mean"
            }).to_dict()
            
        # Per-language statistics
        if "language" in results and "overall_safety_score" in results:
            summary["per_language_stats"] = results.groupby("language").agg({
                "overall_safety_score": ["mean", "std"],
                "is_safe": "mean"
            }).to_dict()
            
        # Per-category statistics
        if "category" in results and "overall_safety_score" in results:
            summary["per_category_stats"] = results.groupby("category").agg({
                "overall_safety_score": ["mean", "std"],
                "is_safe": "mean"
            }).to_dict()
            
        return summary