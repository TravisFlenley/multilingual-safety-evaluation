"""
FastAPI application for the safety evaluation API.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from loguru import logger
from datetime import datetime
import asyncio
from pathlib import Path

from ..core import SafetyEvaluator
from ..utils import get_config_manager


# Pydantic models for API
class EvaluationRequest(BaseModel):
    """Request model for single evaluation."""
    prompt: str = Field(..., description="The prompt to evaluate")
    language: str = Field("en", description="Language code")
    model: Optional[str] = Field(None, description="Model to use for evaluation")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Tell me about AI safety",
                "language": "en",
                "model": "gpt-3.5-turbo"
            }
        }


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation."""
    prompts: List[str] = Field(..., description="List of prompts to evaluate")
    languages: Optional[List[str]] = Field(None, description="List of language codes")
    models: Optional[List[str]] = Field(None, description="List of models to evaluate")
    
    class Config:
        schema_extra = {
            "example": {
                "prompts": ["Tell me about AI safety", "Explain machine learning"],
                "languages": ["en", "es", "zh"],
                "models": ["gpt-3.5-turbo", "claude-3-sonnet"]
            }
        }


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    prompt: str
    language: str
    model: str
    response: str
    overall_safety_score: float
    is_safe: bool
    scores: Dict[str, float]
    flags: Dict[str, bool]
    metadata: Dict[str, Any]
    timestamp: str


class BatchEvaluationResponse(BaseModel):
    """Response model for batch evaluation results."""
    task_id: str
    status: str
    message: str
    result_count: Optional[int] = None
    results_url: Optional[str] = None


# Create FastAPI app
app = FastAPI(
    title="Multilingual Safety Evaluation API",
    description="API for evaluating LLM safety across multiple languages and models",
    version="1.0.0"
)

# Global evaluator instance
evaluator = None

# Background tasks storage
background_tasks_storage = {}


@app.on_event("startup")
async def startup_event():
    """Initialize evaluator on startup."""
    global evaluator
    
    config_manager = get_config_manager()
    evaluator = SafetyEvaluator(config_manager.config_path)
    
    logger.info("Safety Evaluation API started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Multilingual Safety Evaluation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "evaluate": "/api/v1/evaluate",
            "batch_evaluate": "/api/v1/batch-evaluate",
            "models": "/api/v1/models",
            "languages": "/api/v1/languages",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/models")
async def list_models():
    """List available models."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    return {
        "models": evaluator.model_registry.list_models(),
        "count": len(evaluator.model_registry.list_models())
    }


@app.get("/api/v1/languages")
async def list_languages():
    """List supported languages."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    languages = evaluator.config_manager.get_supported_languages()
    
    return {
        "languages": languages,
        "count": len(languages)
    }


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_prompt(request: EvaluationRequest):
    """Evaluate a single prompt."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    try:
        # Perform evaluation
        result = evaluator.evaluate_prompt(
            prompt=request.prompt,
            language=request.language,
            model=request.model
        )
        
        # Convert to response model
        return EvaluationResponse(
            prompt=result.prompt,
            language=result.language,
            model=result.model,
            response=result.response,
            overall_safety_score=result.overall_safety_score(),
            is_safe=result.is_safe(),
            scores=result.scores,
            flags=result.flags,
            metadata=result.metadata,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch-evaluate", response_model=BatchEvaluationResponse)
async def batch_evaluate(request: BatchEvaluationRequest, background_tasks: BackgroundTasks):
    """Start batch evaluation task."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    # Generate task ID
    task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(request.prompts)}"
    
    # Start background task
    background_tasks.add_task(
        run_batch_evaluation,
        task_id,
        request.prompts,
        request.languages,
        request.models
    )
    
    # Store task info
    background_tasks_storage[task_id] = {
        "status": "processing",
        "start_time": datetime.now().isoformat(),
        "prompt_count": len(request.prompts)
    }
    
    return BatchEvaluationResponse(
        task_id=task_id,
        status="processing",
        message=f"Batch evaluation started for {len(request.prompts)} prompts"
    )


async def run_batch_evaluation(task_id: str, prompts: List[str], 
                             languages: Optional[List[str]], 
                             models: Optional[List[str]]):
    """Run batch evaluation in background."""
    try:
        # Create dataset from prompts
        dataset = [{"prompt": prompt} for prompt in prompts]
        
        # Run evaluation
        results = await evaluator.batch_evaluate_async(
            dataset=dataset,
            languages=languages,
            models=models
        )
        
        # Save results
        output_path = Path("data/results") / f"{task_id}.parquet"
        results.to_parquet(output_path)
        
        # Update task status
        background_tasks_storage[task_id] = {
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result_count": len(results),
            "results_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}")
        background_tasks_storage[task_id] = {
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }


@app.get("/api/v1/batch-evaluate/{task_id}")
async def get_batch_status(task_id: str):
    """Get batch evaluation task status."""
    if task_id not in background_tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = background_tasks_storage[task_id]
    
    response = BatchEvaluationResponse(
        task_id=task_id,
        status=task_info["status"],
        message=f"Task is {task_info['status']}"
    )
    
    if task_info["status"] == "completed":
        response.result_count = task_info.get("result_count")
        response.results_url = f"/api/v1/results/{task_id}"
        
    return response


@app.get("/api/v1/results/{task_id}")
async def get_results(task_id: str):
    """Download evaluation results."""
    if task_id not in background_tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = background_tasks_storage[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Results not ready")
        
    results_path = task_info.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(status_code=404, detail="Results file not found")
        
    return FileResponse(
        results_path,
        media_type="application/octet-stream",
        filename=f"{task_id}_results.parquet"
    )


@app.post("/api/v1/compare-models")
async def compare_models(prompts: List[str], models: Optional[List[str]] = None, language: str = "en"):
    """Compare multiple models on the same prompts."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    try:
        comparison_results = evaluator.compare_models(
            prompts=prompts,
            models=models,
            language=language
        )
        
        return {
            "comparison": comparison_results.to_dict(orient="records"),
            "prompt_count": len(prompts),
            "model_count": len(models) if models else len(evaluator.model_registry.list_models())
        }
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/statistics")
async def get_statistics():
    """Get evaluation statistics."""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
        
    try:
        stats = evaluator.get_summary_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()