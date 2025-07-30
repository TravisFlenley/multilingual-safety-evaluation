# API Usage Guide

This guide demonstrates how to use the Multilingual Safety Evaluation API.

## Starting the API Server

First, start the API server:

```bash
python -m src.api.app
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### List Available Models
```bash
curl http://localhost:8000/api/v1/models
```

### List Supported Languages
```bash
curl http://localhost:8000/api/v1/languages
```

### Evaluate a Single Prompt
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about AI safety",
    "language": "en",
    "model": "gpt-3.5-turbo"
  }'
```

### Batch Evaluation
```bash
curl -X POST http://localhost:8000/api/v1/batch-evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "What is machine learning?",
      "Explain neural networks",
      "How does AI work?"
    ],
    "languages": ["en", "es"],
    "models": ["gpt-3.5-turbo"]
  }'
```

### Check Batch Status
```bash
curl http://localhost:8000/api/v1/batch-evaluate/{task_id}
```

### Compare Models
```bash
curl -X POST http://localhost:8000/api/v1/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Explain AI ethics"],
    "language": "en"
  }'
```

## Python Client Usage

### Basic Usage

```python
from src.api.client import SafetyEvalClient

# Initialize client
client = SafetyEvalClient(base_url="http://localhost:8000")

# Evaluate a prompt
result = client.evaluate(
    prompt="Tell me about renewable energy",
    language="en",
    model="gpt-3.5-turbo"
)

print(f"Safety Score: {result['overall_safety_score']}")
print(f"Is Safe: {result['is_safe']}")
```

### Batch Evaluation

```python
# Evaluate multiple prompts
results = client.batch_evaluate(
    prompts=[
        "What is AI?",
        "How does ML work?",
        "Explain deep learning"
    ],
    languages=["en", "es"],
    models=["gpt-3.5-turbo"],
    wait_for_results=True
)

print(f"Results saved to: {results['results_path']}")
```

### Async Usage

```python
import asyncio
from src.api.client import AsyncSafetyEvalClient

async def evaluate_async():
    client = AsyncSafetyEvalClient()
    
    # Evaluate multiple prompts concurrently
    tasks = [
        client.evaluate_async("What is AI?", "en"),
        client.evaluate_async("¿Qué es IA?", "es"),
        client.evaluate_async("什么是人工智能？", "zh")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(f"Score: {result['overall_safety_score']}")

# Run async evaluation
asyncio.run(evaluate_async())
```

## Response Format

### Evaluation Response

```json
{
  "prompt": "Tell me about AI safety",
  "language": "en",
  "model": "gpt-3.5-turbo",
  "response": "AI safety is a critical field...",
  "overall_safety_score": 0.925,
  "is_safe": true,
  "scores": {
    "harmful_content": 0.95,
    "bias": 0.88,
    "privacy": 0.92,
    "toxicity": 0.94
  },
  "flags": {
    "harmful_content_flagged": false,
    "bias_flagged": false,
    "privacy_flagged": false,
    "toxicity_flagged": false
  },
  "metadata": {
    "evaluator": "CompositeEvaluator",
    "latency": 1.23
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### Batch Evaluation Response

```json
{
  "task_id": "batch_20240115_103000_5",
  "status": "completed",
  "message": "Batch evaluation completed",
  "result_count": 15,
  "results_url": "/api/v1/results/batch_20240115_103000_5"
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource not found)
- `500`: Internal Server Error
- `503`: Service Unavailable (evaluator not ready)

Error response format:

```json
{
  "detail": "Error message",
  "error": "Detailed error information"
}
```

## Rate Limiting

The API includes built-in rate limiting (configurable in `config.yaml`):
- Default: 100 requests per minute per IP
- Batch operations: 10 requests per minute per IP

## Authentication

For production use, configure API authentication in `config.yaml`:

```yaml
api_server:
  auth_enabled: true
  api_keys:
    - "your-api-key-here"
```

Then include the API key in requests:

```bash
curl -H "Authorization: Bearer your-api-key-here" \
  http://localhost:8000/api/v1/evaluate
```