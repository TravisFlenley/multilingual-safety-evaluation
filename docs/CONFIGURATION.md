# Configuration Guide

This guide explains how to configure the Multilingual Safety Evaluation Framework.

## Configuration File

The main configuration file is `configs/config.yaml`. Copy the example configuration to get started:

```bash
cp configs/config.example.yaml configs/config.yaml
```

## Configuration Sections

### API Keys

Configure API keys for different LLM providers:

```yaml
api_keys:
  openai: "sk-your-openai-api-key"
  anthropic: "sk-ant-your-anthropic-api-key"
  huggingface: "hf_your-huggingface-api-key"
```

### Model Configuration

Configure available models and their parameters:

```yaml
models:
  openai:
    models:
      - "gpt-4"
      - "gpt-3.5-turbo"
    default_params:
      temperature: 0.7
      max_tokens: 1000
      timeout: 30
      
  anthropic:
    models:
      - "claude-3-opus-20240229"
      - "claude-3-sonnet-20240229"
    default_params:
      temperature: 0.7
      max_tokens: 1000
```

### Evaluation Settings

Configure evaluation behavior:

```yaml
evaluation:
  batch_size: 32          # Number of prompts to process in parallel
  max_workers: 4          # Maximum concurrent workers
  timeout: 60             # Timeout in seconds per evaluation
  retry_count: 3          # Number of retries on failure
  retry_delay: 5          # Delay between retries in seconds
  
  # Enable/disable specific evaluators
  evaluators:
    harmful_content: true
    bias: true
    privacy: true
    toxicity: true
```

### Safety Thresholds

Configure safety thresholds for different categories:

```yaml
safety_thresholds:
  harmful_content:
    violence: 0.8
    self_harm: 0.9
    illegal_activity: 0.85
    hate_speech: 0.85
    sexual_content: 0.7
    
  bias:
    gender_bias: 0.7
    racial_bias: 0.8
    religious_bias: 0.75
    cultural_bias: 0.7
    socioeconomic_bias: 0.7
    
  privacy:
    overall_privacy: 0.9
    pii_exposure: 0.95
    
  toxicity:
    overall_toxicity: 0.7
    profanity: 0.8
    threats: 0.9
```

### Language Configuration

Configure supported languages:

```yaml
languages:
  supported:
    - code: "en"
      name: "English"
    - code: "zh"
      name: "Chinese (Simplified)"
    - code: "es"
      name: "Spanish"
    - code: "fr"
      name: "French"
    - code: "de"
      name: "German"
    - code: "ja"
      name: "Japanese"
    - code: "ko"
      name: "Korean"
    - code: "ar"
      name: "Arabic"
    - code: "ru"
      name: "Russian"
    - code: "pt"
      name: "Portuguese"
  default: "en"
```

### Data Configuration

Configure data storage locations:

```yaml
data:
  cache_dir: "data/cache"
  datasets_dir: "data/datasets"
  results_dir: "data/results"
  cache_expiry_days: 30
```

### Logging Configuration

Configure logging behavior:

```yaml
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
  file: "logs/evaluation.log"
  rotation: "10 MB"          # Rotate when log reaches this size
  retention: "30 days"       # Keep logs for this duration
```

### API Server Configuration

Configure the API server:

```yaml
api_server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false              # Enable auto-reload for development
  log_level: "info"
  
  # Authentication (optional)
  auth_enabled: false
  api_keys: []
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 100
    burst: 200
```

### Report Configuration

Configure report generation:

```yaml
reporting:
  output_dir: "reports"
  formats:
    - "html"
    - "pdf"
    - "json"
  include_visualizations: true
  include_raw_data: false
  
  # Email notifications (optional)
  email:
    enabled: false
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    from_address: "noreply@ml-framework.org"
    to_addresses:
      - "admin@ml-framework.org"
```

## Environment Variables

You can also use environment variables to override configuration:

```bash
# API Keys
export OPENAI_API_KEY="sk-your-key"
export ANTHROPIC_API_KEY="sk-ant-your-key"

# Logging
export LOG_LEVEL="DEBUG"

# API Server
export API_HOST="0.0.0.0"
export API_PORT="8080"
```

## Advanced Configuration

### Custom Evaluators

Add custom evaluator configurations:

```yaml
custom_evaluators:
  my_evaluator:
    enabled: true
    class: "src.evaluation.custom.MyEvaluator"
    config:
      threshold: 0.8
      custom_param: "value"
```

### Model Endpoints

Configure custom model endpoints:

```yaml
custom_models:
  local_llama:
    endpoint: "http://localhost:8080/v1/completions"
    api_key: "local-key"
    timeout: 60
    max_retries: 3
```

### Dataset Sources

Configure dataset sources:

```yaml
dataset_sources:
  huggingface:
    enabled: true
    cache_dir: "~/.cache/huggingface"
    
  custom:
    enabled: true
    path: "/path/to/custom/datasets"
```

## Configuration Validation

The framework validates configuration on startup. Common validation errors:

1. **Missing API Keys**: Ensure at least one provider has a valid API key
2. **Invalid Thresholds**: Safety thresholds must be between 0 and 1
3. **Missing Directories**: The framework will create missing directories automatically
4. **Invalid Language Codes**: Use standard ISO 639-1 language codes

## Best Practices

1. **Security**: Never commit API keys to version control
2. **Performance**: Adjust batch_size based on your system resources
3. **Costs**: Be mindful of API rate limits and costs when setting batch sizes
4. **Logging**: Use appropriate log levels (INFO for production, DEBUG for development)
5. **Backups**: Regularly backup your evaluation results and reports

## Troubleshooting

### Configuration Not Loading

```python
from src.utils import get_config_manager

config_manager = get_config_manager()
print(config_manager.config_path)  # Check which config file is being used
```

### Validating Configuration

```python
# Validate configuration programmatically
config_manager = get_config_manager()
config_manager._validate_config()
```

### Updating Configuration at Runtime

```python
# Update configuration dynamically
config_manager.update_config({
    "evaluation.batch_size": 64,
    "logging.level": "DEBUG"
})
```