"""
Example of using the Safety Evaluation API client.
"""

import asyncio
from src.api.client import SafetyEvalClient, AsyncSafetyEvalClient
import json


def basic_api_example():
    """Basic example of using the API client."""
    
    # Initialize client
    client = SafetyEvalClient(base_url="http://localhost:8000")
    
    # Check if API is running
    print("Checking API health...")
    if not client.health_check():
        print("API is not running. Please start the server with:")
        print("  python -m src.api.app")
        return
    
    print("API is healthy!\n")
    
    # List available models
    print("=== Available Models ===")
    models = client.list_models()
    for model in models:
        print(f"  - {model}")
    
    # List supported languages
    print("\n=== Supported Languages ===")
    languages = client.list_languages()
    for lang in languages:
        print(f"  - {lang}")
    
    # Evaluate a single prompt
    print("\n=== Single Prompt Evaluation ===")
    result = client.evaluate(
        prompt="Tell me about the benefits of renewable energy",
        language="en"
    )
    
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response'][:100]}...")
    print(f"Safety Score: {result['overall_safety_score']:.3f}")
    print(f"Is Safe: {result['is_safe']}")
    
    # Pretty print scores
    print("\nDetailed Scores:")
    for category, score in result['scores'].items():
        print(f"  {category}: {score:.3f}")


def batch_evaluation_example():
    """Example of batch evaluation via API."""
    
    client = SafetyEvalClient()
    
    print("\n=== Batch Evaluation Example ===")
    
    # Prepare test prompts
    prompts = [
        "What is machine learning?",
        "How does natural language processing work?",
        "Explain deep learning concepts",
        "What are the applications of AI in healthcare?",
        "How can AI help with climate change?"
    ]
    
    # Submit batch evaluation
    print(f"Submitting {len(prompts)} prompts for evaluation...")
    result = client.batch_evaluate(
        prompts=prompts,
        languages=["en", "es"],
        models=["gpt-3.5-turbo"],
        wait_for_results=True,
        timeout=300
    )
    
    print(f"Batch evaluation completed!")
    print(f"Task ID: {result['task_id']}")
    print(f"Results saved to: {result.get('results_path', 'N/A')}")


def model_comparison_api_example():
    """Example of comparing models via API."""
    
    client = SafetyEvalClient()
    
    print("\n=== Model Comparison via API ===")
    
    # Test prompts for comparison
    prompts = [
        "Explain the concept of privacy in the digital age",
        "What are the ethical considerations in AI development?",
        "How can we ensure fairness in machine learning?"
    ]
    
    # Compare models
    comparison = client.compare_models(
        prompts=prompts,
        models=None,  # Use all available models
        language="en"
    )
    
    print(f"\nCompared {len(comparison)} prompt-model combinations")
    
    # Display results
    for i, result in enumerate(comparison[:3]):  # Show first 3
        print(f"\nPrompt: {result['prompt'][:50]}...")
        for key, value in result.items():
            if key.endswith("_safety_score"):
                model = key.replace("_safety_score", "")
                print(f"  {model}: {value:.3f}")


async def async_api_example():
    """Example of using async API client."""
    
    client = AsyncSafetyEvalClient()
    
    print("\n=== Async API Example ===")
    
    # Evaluate multiple prompts concurrently
    prompts = [
        ("What is quantum computing?", "en"),
        ("¿Qué es la computación cuántica?", "es"),
        ("什么是量子计算？", "zh"),
        ("Qu'est-ce que l'informatique quantique?", "fr"),
        ("Was ist Quantencomputing?", "de")
    ]
    
    # Create evaluation tasks
    tasks = []
    for prompt, language in prompts:
        task = client.evaluate_async(prompt=prompt, language=language)
        tasks.append(task)
    
    # Run all evaluations concurrently
    print(f"Evaluating {len(prompts)} prompts concurrently...")
    results = await asyncio.gather(*tasks)
    
    # Display results
    for (prompt, lang), result in zip(prompts, results):
        print(f"\n[{lang}] {prompt[:30]}...")
        print(f"  Safety Score: {result['overall_safety_score']:.3f}")
        print(f"  Is Safe: {result['is_safe']}")


def statistics_example():
    """Example of retrieving evaluation statistics."""
    
    client = SafetyEvalClient()
    
    print("\n=== Evaluation Statistics ===")
    
    # First, run some evaluations to generate statistics
    test_prompts = [
        "Tell me about AI safety",
        "What are neural networks?",
        "How does machine learning work?"
    ]
    
    for prompt in test_prompts:
        client.evaluate(prompt=prompt, language="en")
    
    # Get statistics
    stats = client.get_statistics()
    
    print("\nCurrent Statistics:")
    print(f"Total Evaluations: {stats.get('total_evaluations', 0)}")
    print(f"Models Evaluated: {stats.get('models_evaluated', 0)}")
    print(f"Overall Safety Rate: {stats.get('overall_safety_rate', 0):.2%}")
    print(f"Average Safety Score: {stats.get('average_safety_score', 0):.3f}")
    
    # Pretty print per-model stats if available
    if 'per_model_stats' in stats:
        print("\nPer-Model Statistics:")
        model_stats = stats['per_model_stats']
        
        # The structure is nested, so we need to parse it carefully
        if 'overall_safety_score' in model_stats:
            for stat_type in ['mean', 'std', 'min', 'max']:
                if stat_type in model_stats['overall_safety_score']:
                    print(f"\n  {stat_type.capitalize()} Safety Scores:")
                    for model, value in model_stats['overall_safety_score'][stat_type].items():
                        print(f"    {model}: {value:.3f}")


def main():
    """Run all API examples."""
    
    print("=== Safety Evaluation API Client Examples ===\n")
    
    # Basic example
    basic_api_example()
    
    # Batch evaluation
    batch_evaluation_example()
    
    # Model comparison
    model_comparison_api_example()
    
    # Statistics
    statistics_example()
    
    # Async example (uncomment to run)
    # print("\nRunning async example...")
    # asyncio.run(async_api_example())
    
    print("\n=== API examples completed! ===")


if __name__ == "__main__":
    main()