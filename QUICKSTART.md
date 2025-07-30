# Quick Start Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning the repository)

### Quick Setup

1. **Run the quick setup script:**
   ```bash
   python quickstart.py
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies
   - Set up configuration files
   - Create necessary directories
   - Run basic tests

2. **Configure API Keys:**
   Edit `configs/config.yaml` and add your API keys:
   ```yaml
   api_keys:
     openai: "sk-your-openai-api-key"
     anthropic: "sk-ant-your-anthropic-api-key"
   ```

3. **Activate Virtual Environment:**
   ```bash
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

## Basic Usage

### Command Line Interface

1. **Evaluate a single prompt:**
   ```bash
   ml-safety-eval evaluate -p "Tell me about AI safety" -l en
   ```

2. **Batch evaluation:**
   ```bash
   ml-safety-eval batch -f prompts.csv -o results.csv
   ```

3. **Compare models:**
   ```bash
   ml-safety-eval compare -p "Explain ML" -p "What is AI?" -m gpt-3.5-turbo -m claude-3
   ```

4. **Generate report:**
   ```bash
   ml-safety-eval report -r results.csv -f html -f json
   ```

### Python API

```python
from src.core import SafetyEvaluator

# Initialize evaluator
evaluator = SafetyEvaluator()

# Evaluate single prompt
result = evaluator.evaluate_prompt(
    prompt="What is machine learning?",
    language="en"
)

print(f"Safety Score: {result.overall_safety_score():.3f}")
print(f"Is Safe: {result.is_safe()}")
```

### REST API

1. **Start the API server:**
   ```bash
   ml-safety-api
   # or
   python -m src.api.app
   ```

2. **Make requests:**
   ```bash
   # Evaluate prompt
   curl -X POST http://localhost:8000/api/v1/evaluate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Tell me about AI",
       "language": "en"
     }'
   ```

## Examples

Run the example scripts to see the framework in action:

```bash
# Basic examples
python examples/basic_usage.py

# Advanced examples
python examples/advanced_usage.py

# API client examples
python examples/api_usage.py
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src tests/
```

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md) - Detailed configuration options
- [API Guide](docs/API_GUIDE.md) - REST API documentation
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

## Common Issues

### Missing API Keys
If you see errors about missing API keys:
1. Check that `configs/config.yaml` exists
2. Ensure your API keys are correctly formatted
3. Verify the keys are valid with the providers

### Import Errors
If you get import errors:
1. Make sure the virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check that you're in the project root directory

### API Connection Issues
If the API won't start:
1. Check that port 8000 is not in use
2. Try a different port: `ml-safety-api --port 8080`
3. Check firewall settings

## Support

For issues and questions:
- Check the [documentation](docs/)
- Open an issue on [GitHub](https://github.com/TravisFlenley/multilingual-safety-evaluation)
- Contact: olmate1@gmail.com
