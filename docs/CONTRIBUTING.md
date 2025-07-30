# Contributing to Multilingual Safety Evaluation Framework

We welcome contributions to the Multilingual Safety Evaluation Framework! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/multilingual-safety-evaluation-optimized.git
   cd multilingual-safety-evaluation-optimized
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Setup

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/unit/test_evaluator.py
```

Run with coverage:
```bash
pytest --cov=src tests/
```

### Code Style

We follow PEP 8 and use the following tools:
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

Format code:
```bash
black src/ tests/
```

Check linting:
```bash
flake8 src/ tests/
```

Type checking:
```bash
mypy src/
```

## Contributing Guidelines

### Reporting Issues

1. Check if the issue already exists
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - System information

### Submitting Pull Requests

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes:
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `style:` formatting changes
   - `refactor:` code refactoring
   - `test:` test changes
   - `chore:` maintenance tasks

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a Pull Request

### Pull Request Guidelines

- **Title**: Use conventional commit format
- **Description**: Explain what changes you made and why
- **Tests**: Ensure all tests pass
- **Documentation**: Update relevant documentation
- **Code Review**: Address reviewer feedback promptly

## Development Areas

### Adding New Evaluators

1. Create a new evaluator in `src/evaluation/evaluators/`:
   ```python
   from ..base_evaluator import BaseEvaluator
   
   class MyEvaluator(BaseEvaluator):
       def evaluate(self, prompt, response, language="en", **kwargs):
           # Implementation
           pass
   ```

2. Add tests in `tests/unit/test_evaluators.py`

3. Register in `src/evaluation/__init__.py`

4. Update documentation

### Adding Model Support

1. Create model interface in `src/models/`:
   ```python
   from .base_model import BaseModel
   
   class MyModel(BaseModel):
       def generate(self, prompt, **kwargs):
           # Implementation
           pass
   ```

2. Add configuration in `configs/config.yaml`

3. Add tests and documentation

### Adding Language Support

1. Update language configuration
2. Add language-specific patterns in evaluators
3. Add test cases for the new language
4. Update documentation

## Testing Guidelines

### Unit Tests

Test individual components:
```python
def test_evaluator_initialization():
    evaluator = HarmfulContentEvaluator()
    assert evaluator is not None
    
def test_evaluation_logic():
    evaluator = HarmfulContentEvaluator()
    result = evaluator.evaluate("test prompt", "test response")
    assert "scores" in result
    assert "flags" in result
```

### Integration Tests

Test component interactions:
```python
def test_full_evaluation_pipeline():
    evaluator = SafetyEvaluator()
    result = evaluator.evaluate_prompt("test prompt")
    assert result.is_safe() in [True, False]
```

### Performance Tests

Test performance and scalability:
```python
def test_batch_evaluation_performance():
    evaluator = SafetyEvaluator()
    prompts = ["test"] * 100
    
    start_time = time.time()
    results = evaluator.batch_evaluate(prompts)
    duration = time.time() - start_time
    
    assert duration < 60  # Should complete within 60 seconds
```

## Documentation

### Code Documentation

Use clear docstrings:
```python
def evaluate(self, prompt: str, response: str, language: str = "en") -> Dict[str, Any]:
    """
    Evaluate a prompt-response pair for safety.
    
    Args:
        prompt: The input prompt
        response: The model's response
        language: Language code (ISO 639-1)
        
    Returns:
        Dictionary containing scores, flags, and metadata
        
    Raises:
        ValueError: If language is not supported
    """
```

### API Documentation

Update API documentation when adding endpoints:
- Add to `docs/API_GUIDE.md`
- Include request/response examples
- Document error cases

### User Documentation

Update user guides for new features:
- Add examples in `examples/`
- Update README.md
- Add to relevant guides in `docs/`

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Build documentation
6. Tag release
7. Create GitHub release

## Community

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful and inclusive in all interactions.

### Getting Help

- Create an issue for bugs or feature requests
- Join our Discord community (link in README)
- Check existing documentation and issues

### Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## Development Tips

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Profile code performance:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
evaluator.batch_evaluate(dataset)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Profiling

Check memory usage:
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

Thank you for contributing to the Multilingual Safety Evaluation Framework!