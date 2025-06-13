"""
Command-line interface for household energy segmentation.

This module provides a simple CLI for running clustering experiments
and evaluations.
"""

import click
import sys
from pathlib import Path

from .utils.config import load_config
from .utils.logging import setup_logging, get_logger
from .utils.helpers import set_random_seed


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Household Energy Segmentation CLI."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=log_level)
    
    # Load configuration
    ctx.obj['config'] = load_config(config)
    ctx.obj['logger'] = get_logger(__name__)


@cli.command()
@click.option('--method', '-m', 
              type=click.Choice(['sax_kmeans', 'two_stage_kmeans', 'dec', 'weather_dec']),
              default='weather_dec',
              help='Clustering method to use')
@click.option('--data-path', '-d', type=click.Path(), help='Path to data file')
@click.option('--output-dir', '-o', type=click.Path(), default='results', help='Output directory')
@click.option('--n-clusters', '-k', type=int, default=3, help='Number of clusters')
@click.pass_context
def cluster(ctx, method, data_path, output_dir, n_clusters):
    """Run clustering on energy consumption data."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    logger.info(f"Running {method} clustering with {n_clusters} clusters")
    
    # Set random seed
    set_random_seed(config.model.random_state if hasattr(config.model, 'random_state') else 42)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if method == 'weather_dec':
            from .examples.comprehensive_example import main as run_example
            results = run_example()
            logger.info("Weather-fused DEC clustering completed successfully")
        else:
            logger.error(f"Method {method} not yet implemented in CLI")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--n-samples', '-n', type=int, default=100, help='Number of samples to generate')
@click.option('--n-clusters', '-k', type=int, default=3, help='Number of clusters')
@click.option('--output-path', '-o', type=click.Path(), default='synthetic_data.npy', help='Output file path')
@click.pass_context
def generate_data(ctx, n_samples, n_clusters, output_path):
    """Generate synthetic data for testing."""
    logger = ctx.obj['logger']
    
    logger.info(f"Generating {n_samples} synthetic samples with {n_clusters} clusters")
    
    try:
        from .data.synthetic import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(random_state=42)
        load_shapes, weather_data, true_labels = generator.generate_clustered_data(
            n_samples=n_samples,
            n_clusters=n_clusters
        )
        
        # Save data
        import numpy as np
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            load_shapes=load_shapes,
            weather_data=weather_data,
            true_labels=true_labels
        )
        
        logger.info(f"Synthetic data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--results-path', '-r', type=click.Path(exists=True), 
              help='Path to evaluation results JSON file')
@click.pass_context
def report(ctx, results_path):
    """Generate evaluation report from results."""
    logger = ctx.obj['logger']
    
    if not results_path:
        logger.error("Results path is required")
        sys.exit(1)
        
    try:
        from .evaluation.evaluator import ClusteringEvaluator
        
        evaluator = ClusteringEvaluator()
        evaluator.load_results(results_path)
        
        # Generate reports for all methods
        for method_name in evaluator.results.keys():
            report_path = Path(results_path).parent / f"{method_name}_report.txt"
            report = evaluator.generate_report(method_name, report_path)
            logger.info(f"Report generated for {method_name}: {report_path}")
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display system and package information."""
    logger = ctx.obj['logger']
    
    try:
        from .utils.helpers import print_system_info
        print_system_info()
        
    except Exception as e:
        logger.error(f"Failed to display system info: {e}")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
