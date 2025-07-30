# Project Structure

```
multilingual-safety-evaluation-optimized/
├── src/                        # Source code
│   ├── __init__.py
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── safety_evaluator.py    # Main evaluator class
│   │   └── report_generator.py    # Report generation
│   │
│   ├── evaluation/             # Evaluation modules
│   │   ├── __init__.py
│   │   ├── base_evaluator.py      # Base evaluator classes
│   │   └── safety_evaluators.py   # Safety evaluators
│   │
│   ├── models/                 # Model interfaces
│   │   ├── __init__.py
│   │   ├── base_model.py          # Base model interface
│   │   ├── openai_model.py        # OpenAI integration
│   │   └── anthropic_model.py     # Anthropic integration
│   │
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   └── dataset_collector.py   # Dataset management
│   │
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   └── config_manager.py     # Configuration management
│   │
│   ├── api/                    # API layer
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   └── client.py              # API client
│   │
│   ├── safety/                 # Safety specific modules
│   │   └── __init__.py
│   │
│   └── cli.py                  # Command-line interface
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Test configuration
│   ├── unit/                      # Unit tests
│   │   ├── __init__.py
│   │   ├── test_evaluators.py
│   │   ├── test_safety_evaluators.py
│   │   ├── test_models.py
│   │   └── test_config.py
│   │
│   └── integration/            # Integration tests
│       ├── __init__.py
│       └── test_integration.py
│
├── configs/                    # Configuration files
│   ├── config.yaml                # Main configuration
│   └── config.example.yaml        # Example configuration
│
├── data/                       # Data directory
│   ├── datasets/                  # Evaluation datasets
│   ├── cache/                     # Cache directory
│   └── results/                   # Results storage
│
├── docs/                       # Documentation
│   ├── API_GUIDE.md              # API documentation
│   ├── CONFIGURATION.md          # Configuration guide
│   └── CONTRIBUTING.md           # Contributing guide
│
├── examples/                   # Example scripts
│   ├── basic_usage.py            # Basic examples
│   ├── advanced_usage.py         # Advanced examples
│   └── api_usage.py              # API examples
│
├── logs/                       # Log files
├── reports/                    # Generated reports
├── scripts/                    # Utility scripts
│
├── README.md                   # Project documentation
├── QUICKSTART.md              # Quick start guide
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package setup
├── quickstart.py              # Quick setup script
└── .gitignore                 # Git ignore file
```

## Key Components

### Core (`src/core/`)
- **SafetyEvaluator**: Main orchestrator for safety evaluations
- **ReportGenerator**: Generates HTML/JSON reports with visualizations

### Evaluation (`src/evaluation/`)
- **BaseEvaluator**: Abstract base class for all evaluators
- **CompositeEvaluator**: Combines multiple evaluators
- **HarmfulContentEvaluator**: Detects harmful/dangerous content
- **BiasEvaluator**: Identifies various types of bias
- **PrivacyEvaluator**: Checks for PII and privacy violations
- **ToxicityEvaluator**: Detects toxic language

### Models (`src/models/`)
- **BaseModel**: Abstract interface for LLM providers
- **ModelRegistry**: Manages multiple model instances
- **OpenAIModel**: OpenAI GPT models integration
- **AnthropicModel**: Anthropic Claude models integration

### Data (`src/data/`)
- **DatasetCollector**: Collects and prepares evaluation datasets
- **DatasetManager**: High-level dataset management

### API (`src/api/`)
- **FastAPI Application**: RESTful API server
- **API Client**: Python client for API interaction

### Utilities (`src/utils/`)
- **ConfigManager**: Configuration loading and validation
- **Config**: Configuration object with dot notation access

## Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new evaluators, models, or languages
3. **Type Safety**: Uses type hints throughout
4. **Async Support**: Supports both sync and async operations
5. **Configuration-Driven**: Behavior controlled via configuration
6. **Comprehensive Testing**: Unit and integration tests
7. **Documentation**: Extensive documentation and examples