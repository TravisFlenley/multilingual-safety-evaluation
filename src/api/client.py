"""
API client for interacting with the safety evaluation API.
"""

import httpx
from typing import List, Optional, Dict, Any
import asyncio
from loguru import logger
import time
from pathlib import Path


class SafetyEvalClient:
    """Client for the Safety Evaluation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
        # Setup headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            
    def evaluate(self, prompt: str, language: str = "en", 
                model: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single prompt.
        
        Args:
            prompt: Prompt to evaluate
            language: Language code
            model: Model to use
            
        Returns:
            Evaluation results
        """
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/api/v1/evaluate",
                json={
                    "prompt": prompt,
                    "language": language,
                    "model": model
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
    async def evaluate_async(self, prompt: str, language: str = "en",
                           model: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single prompt asynchronously.
        
        Args:
            prompt: Prompt to evaluate
            language: Language code
            model: Model to use
            
        Returns:
            Evaluation results
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/evaluate",
                json={
                    "prompt": prompt,
                    "language": language,
                    "model": model
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
    def batch_evaluate(self, prompts: List[str], 
                      languages: Optional[List[str]] = None,
                      models: Optional[List[str]] = None,
                      wait_for_results: bool = True,
                      timeout: int = 300) -> Dict[str, Any]:
        """
        Evaluate multiple prompts.
        
        Args:
            prompts: List of prompts
            languages: List of languages
            models: List of models
            wait_for_results: Whether to wait for completion
            timeout: Timeout in seconds
            
        Returns:
            Batch evaluation results or task info
        """
        with httpx.Client() as client:
            # Start batch evaluation
            response = client.post(
                f"{self.base_url}/api/v1/batch-evaluate",
                json={
                    "prompts": prompts,
                    "languages": languages,
                    "models": models
                },
                headers=self.headers
            )
            response.raise_for_status()
            task_info = response.json()
            
            if not wait_for_results:
                return task_info
                
            # Poll for results
            task_id = task_info["task_id"]
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status_response = client.get(
                    f"{self.base_url}/api/v1/batch-evaluate/{task_id}",
                    headers=self.headers
                )
                status_response.raise_for_status()
                status = status_response.json()
                
                if status["status"] == "completed":
                    # Download results
                    results_response = client.get(
                        f"{self.base_url}/api/v1/results/{task_id}",
                        headers=self.headers
                    )
                    results_response.raise_for_status()
                    
                    # Save results
                    output_path = Path(f"results_{task_id}.parquet")
                    with open(output_path, "wb") as f:
                        f.write(results_response.content)
                        
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "results_path": str(output_path)
                    }
                    
                elif status["status"] == "failed":
                    raise Exception(f"Batch evaluation failed: {status}")
                    
                time.sleep(5)  # Poll every 5 seconds
                
            raise TimeoutError(f"Batch evaluation timed out after {timeout} seconds")
            
    def list_models(self) -> List[str]:
        """Get list of available models."""
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/api/v1/models",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["models"]
            
    def list_languages(self) -> List[str]:
        """Get list of supported languages."""
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/api/v1/languages",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["languages"]
            
    def compare_models(self, prompts: List[str], 
                      models: Optional[List[str]] = None,
                      language: str = "en") -> List[Dict[str, Any]]:
        """
        Compare multiple models on the same prompts.
        
        Args:
            prompts: List of prompts
            models: List of models to compare
            language: Language code
            
        Returns:
            Comparison results
        """
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/api/v1/compare-models",
                json={
                    "prompts": prompts,
                    "models": models,
                    "language": language
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["comparison"]
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        with httpx.Client() as client:
            response = client.get(
                f"{self.base_url}/api/v1/statistics",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False


# Async client version
class AsyncSafetyEvalClient(SafetyEvalClient):
    """Async client for the Safety Evaluation API."""
    
    async def batch_evaluate_async(self, prompts: List[str],
                                  languages: Optional[List[str]] = None,
                                  models: Optional[List[str]] = None,
                                  wait_for_results: bool = True,
                                  timeout: int = 300) -> Dict[str, Any]:
        """
        Evaluate multiple prompts asynchronously.
        
        Args:
            prompts: List of prompts
            languages: List of languages
            models: List of models
            wait_for_results: Whether to wait for completion
            timeout: Timeout in seconds
            
        Returns:
            Batch evaluation results or task info
        """
        async with httpx.AsyncClient() as client:
            # Start batch evaluation
            response = await client.post(
                f"{self.base_url}/api/v1/batch-evaluate",
                json={
                    "prompts": prompts,
                    "languages": languages,
                    "models": models
                },
                headers=self.headers
            )
            response.raise_for_status()
            task_info = response.json()
            
            if not wait_for_results:
                return task_info
                
            # Poll for results
            task_id = task_info["task_id"]
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status_response = await client.get(
                    f"{self.base_url}/api/v1/batch-evaluate/{task_id}",
                    headers=self.headers
                )
                status_response.raise_for_status()
                status = status_response.json()
                
                if status["status"] == "completed":
                    return status
                elif status["status"] == "failed":
                    raise Exception(f"Batch evaluation failed: {status}")
                    
                await asyncio.sleep(5)
                
            raise TimeoutError(f"Batch evaluation timed out after {timeout} seconds")


if __name__ == "__main__":
    # Example usage
    client = SafetyEvalClient()
    
    # Check health
    print("API Health:", client.health_check())
    
    # List models
    print("Available models:", client.list_models())
    
    # Evaluate a prompt
    result = client.evaluate(
        prompt="Tell me about AI safety",
        language="en"
    )
    print("Evaluation result:", result)