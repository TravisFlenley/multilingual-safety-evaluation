"""
Command-line interface for the Safety Evaluation Framework.
"""

import click
import json
from pathlib import Path
from loguru import logger
from src.core import SafetyEvaluator
from src.api.app import run_server
from src.data import DatasetManager
import pandas as pd


@click.group()
@click.option('--config', '-c', default='configs/config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """Multilingual Safety Evaluation Framework CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.option('--prompt', '-p', required=True, help='Prompt to evaluate')
@click.option('--language', '-l', default='en', help='Language code')
@click.option('--model', '-m', help='Model to use')
@click.pass_context
def evaluate(ctx, prompt, language, model):
    """Evaluate a single prompt."""
    evaluator = SafetyEvaluator(ctx.obj['config'])
    
    click.echo(f"Evaluating prompt in {language}...")
    result = evaluator.evaluate_prompt(prompt, language, model)
    
    click.echo("\n=== Evaluation Results ===")
    click.echo(f"Prompt: {result.prompt}")
    click.echo(f"Response: {result.response[:200]}...")
    click.echo(f"Overall Safety Score: {result.overall_safety_score():.3f}")
    click.echo(f"Is Safe: {'✓' if result.is_safe() else '✗'}")
    
    click.echo("\nDetailed Scores:")
    for category, score in result.scores.items():
        status = '✓' if score > 0.7 else '✗'
        click.echo(f"  {category}: {score:.3f} {status}")


@cli.command()
@click.option('--file', '-f', required=True, type=click.Path(exists=True), help='Dataset file path')
@click.option('--models', '-m', multiple=True, help='Models to evaluate')
@click.option('--languages', '-l', multiple=True, help='Languages to evaluate')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', type=click.Choice(['csv', 'json', 'parquet']), default='csv', help='Output format')
@click.pass_context
def batch(ctx, file, models, languages, output, format):
    """Run batch evaluation on a dataset."""
    evaluator = SafetyEvaluator(ctx.obj['config'])
    
    click.echo(f"Loading dataset from {file}...")
    
    # Load dataset based on file extension
    file_path = Path(file)
    if file_path.suffix == '.csv':
        dataset = pd.read_csv(file)
    elif file_path.suffix == '.json':
        dataset = pd.read_json(file)
    elif file_path.suffix == '.parquet':
        dataset = pd.read_parquet(file)
    else:
        dataset = pd.read_csv(file)
    
    click.echo(f"Loaded {len(dataset)} prompts")
    
    # Run evaluation
    with click.progressbar(length=len(dataset), label='Evaluating') as bar:
        results = evaluator.batch_evaluate(
            dataset=dataset,
            models=list(models) if models else None,
            languages=list(languages) if languages else None,
            save_results=True
        )
        bar.update(len(dataset))
    
    # Save results
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"evaluation_results.{format}")
        
    if format == 'csv':
        results.to_csv(output_path, index=False)
    elif format == 'json':
        results.to_json(output_path, orient='records', indent=2)
    elif format == 'parquet':
        results.to_parquet(output_path, index=False)
        
    click.echo(f"\nResults saved to {output_path}")
    
    # Show summary
    click.echo("\n=== Summary ===")
    click.echo(f"Total evaluations: {len(results)}")
    click.echo(f"Average safety score: {results['overall_safety_score'].mean():.3f}")
    click.echo(f"Safe responses: {results['is_safe'].sum()}/{len(results)} ({results['is_safe'].mean()*100:.1f}%)")


@cli.command()
@click.option('--results', '-r', type=click.Path(exists=True), help='Results file path')
@click.option('--output', '-o', default='reports', help='Output directory')
@click.option('--formats', '-f', multiple=True, default=['html', 'json'], help='Report formats')
@click.pass_context
def report(ctx, results, output, formats):
    """Generate evaluation report."""
    evaluator = SafetyEvaluator(ctx.obj['config'])
    
    # Load results if provided
    if results:
        click.echo(f"Loading results from {results}...")
        results_path = Path(results)
        if results_path.suffix == '.csv':
            results_df = pd.read_csv(results)
        elif results_path.suffix == '.json':
            results_df = pd.read_json(results)
        elif results_path.suffix == '.parquet':
            results_df = pd.read_parquet(results)
        else:
            results_df = pd.read_csv(results)
    else:
        results_df = None
        
    click.echo("Generating report...")
    report_paths = evaluator.generate_report(
        results=results_df,
        output_path=output,
        formats=list(formats)
    )
    
    click.echo("\n=== Reports Generated ===")
    for fmt, path in report_paths.items():
        click.echo(f"{fmt}: {path}")


@cli.command()
@click.option('--prompts', '-p', multiple=True, required=True, help='Prompts to compare')
@click.option('--models', '-m', multiple=True, help='Models to compare')
@click.option('--language', '-l', default='en', help='Language code')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def compare(ctx, prompts, models, language, output):
    """Compare multiple models on the same prompts."""
    evaluator = SafetyEvaluator(ctx.obj['config'])
    
    click.echo(f"Comparing {len(models) if models else 'all'} models on {len(prompts)} prompts...")
    
    comparison = evaluator.compare_models(
        prompts=list(prompts),
        models=list(models) if models else None,
        language=language
    )
    
    # Display results
    click.echo("\n=== Model Comparison ===")
    for _, row in comparison.iterrows():
        click.echo(f"\nPrompt: {row['prompt'][:50]}...")
        
        # Extract model scores
        model_scores = {}
        for col in comparison.columns:
            if col.endswith('_safety_score'):
                model_name = col.replace('_safety_score', '')
                model_scores[model_name] = row[col]
                
        # Sort by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        for model, score in sorted_models:
            status = '✓' if score > 0.8 else '✗'
            click.echo(f"  {model}: {score:.3f} {status}")
            
    # Save if output specified
    if output:
        comparison.to_csv(output, index=False)
        click.echo(f"\nComparison saved to {output}")


@cli.command()
@click.option('--languages', '-l', multiple=True, help='Languages to prepare data for')
@click.option('--categories', '-c', multiple=True, help='Safety categories to include')
@click.option('--output', '-o', default='data/datasets', help='Output directory')
@click.pass_context
def prepare_data(ctx, languages, categories, output):
    """Prepare evaluation datasets."""
    manager = DatasetManager()
    
    languages = list(languages) if languages else ['en', 'es', 'zh']
    categories = list(categories) if categories else None
    
    click.echo(f"Preparing datasets for languages: {', '.join(languages)}")
    
    datasets = manager.prepare_evaluation_data(
        languages=languages,
        categories=categories
    )
    
    # Export datasets
    click.echo("\nExporting datasets...")
    export_paths = manager.export_for_evaluation(output)
    
    click.echo("\n=== Datasets Created ===")
    stats = manager.get_dataset_stats()
    for _, row in stats.iterrows():
        click.echo(f"{row['language']}: {row['total_samples']} samples, "
                  f"{row['categories']} categories")


@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='API host')
@click.option('--port', '-p', default=8000, type=int, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server."""
    click.echo(f"Starting API server on {host}:{port}")
    run_server(host=host, port=port, reload=reload)


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information."""
    from src.utils import get_config_manager
    
    config_manager = get_config_manager(ctx.obj['config'])
    
    click.echo("=== Multilingual Safety Evaluation Framework ===")
    click.echo(f"Version: 1.0.0")
    click.echo(f"Config: {config_manager.config_path}")
    
    click.echo("\nConfigured Providers:")
    for provider in ['openai', 'anthropic', 'huggingface']:
        api_key = config_manager.get_api_key(provider)
        status = '✓' if api_key else '✗'
        click.echo(f"  {provider}: {status}")
        
    click.echo(f"\nSupported Languages: {', '.join(config_manager.get_supported_languages())}")
    
    click.echo("\nSafety Thresholds:")
    for category in ['harmful_content', 'bias', 'privacy', 'toxicity']:
        threshold = config_manager.get_safety_threshold(category)
        click.echo(f"  {category}: {threshold}")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()