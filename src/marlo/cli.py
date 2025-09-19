"""
Command Line Interface for MARLO Traffic Signal Optimization.
Provides commands for training, evaluation, data collection, and visualization.
"""

import argparse
import sys
import os
import yaml
from typing import Optional, Dict, Any

from .training.trainer import OfflineTrainer
from .training.eval_runner import EvaluationRunner
from .data.dataset_builder import DatasetBuilder
from .viz.plotter import MARLOPlotter, create_dashboard
from .utils.logger import get_logger
from .utils.seed import set_seed


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MARLO: Multi-Agent Reinforcement Learning for Traffic Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.marlo.cli collect-data --config configs/base.yaml --episodes 100
  python -m src.marlo.cli train --config configs/base.yaml
  python -m src.marlo.cli eval --config configs/base.yaml --checkpoint experiments/experiment_01/checkpoints/best_model.pt
  python -m src.marlo.cli visualize --experiment experiment_01
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser(
        'collect-data', 
        help='Generate synthetic datasets for offline training'
    )
    collect_parser.add_argument(
        '--config', 
        type=str, 
        default='configs/base.yaml',
        help='Path to configuration file'
    )
    collect_parser.add_argument(
        '--episodes', 
        type=int, 
        default=100,
        help='Number of episodes to generate'
    )
    collect_parser.add_argument(
        '--dataset-type', 
        type=str, 
        choices=['synthetic', 'semi-synthetic'],
        default='synthetic',
        help='Type of dataset to generate'
    )
    collect_parser.add_argument(
        '--output-path', 
        type=str,
        help='Custom output path for dataset'
    )
    collect_parser.add_argument(
        '--policy', 
        type=str, 
        choices=['random', 'fixed_time', 'greedy', 'adaptive'],
        default='random',
        help='Policy to use for data collection'
    )
    collect_parser.add_argument(
        '--seed', 
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        'train', 
        help='Train multi-agent traffic signal controllers'
    )
    train_parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Compute device override (auto tries CUDA, else CPU)'
    )
    train_parser.add_argument(
        '--resume', 
        type=str,
        help='Path to checkpoint to resume training from'
    )
    train_parser.add_argument(
        '--experiment-name', 
        type=str,
        help='Override experiment name from config'
    )
    
    # Evaluation command
    eval_parser = subparsers.add_parser(
        'eval', 
        help='Evaluate trained agents'
    )
    eval_parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration file'
    )
    eval_parser.add_argument(
        '--checkpoint', 
        type=str,
        help='Path to trained model checkpoint'
    )
    eval_parser.add_argument(
        '--episodes', 
        type=int, 
        default=100,
        help='Number of evaluation episodes'
    )
    eval_parser.add_argument(
        '--deterministic', 
        action='store_true',
        help='Use deterministic policy (no exploration)'
    )
    eval_parser.add_argument(
        '--baseline-comparison', 
        type=str, 
        choices=['random', 'fixed_time'],
        help='Compare with baseline policy'
    )
    eval_parser.add_argument(
        '--render', 
        action='store_true',
        help='Render episodes during evaluation'
    )
    eval_parser.add_argument(
        '--output-file', 
        type=str,
        help='File to save evaluation results'
    )
    eval_parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Compute device override (auto tries CUDA, else CPU)'
    )
    
    # Visualization command
    viz_parser = subparsers.add_parser(
        'visualize', 
        help='Generate visualization plots'
    )
    viz_parser.add_argument(
        '--experiment', 
        type=str, 
        required=True,
        help='Experiment name to visualize'
    )
    viz_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Directory to save plots'
    )
    viz_parser.add_argument(
        '--plots', 
        type=str, 
        nargs='+',
        choices=['learning', 'metrics', 'turns', 'heatmap', 'comparison', 'dashboard'],
        default=['dashboard'],
        help='Specific plots to generate'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'collect-data':
            collect_data_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'eval':
            eval_command(args)
        elif args.command == 'visualize':
            visualize_command(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


def collect_data_command(args):
    """Execute data collection command."""
    logger = get_logger("cli.collect_data")
    logger.info(f"Starting data collection: {args.dataset_type} dataset with {args.episodes} episodes")
    
    # Load config if provided
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set seed if provided
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    
    # Initialize dataset builder
    builder = DatasetBuilder()
    
    # Generate output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = f"datasets/{args.dataset_type}"
        output_file = f"dataset_{args.policy}_{args.episodes}ep.npz"
        output_path = os.path.join(output_dir, output_file)
    
    # Generate dataset
    if args.dataset_type == 'synthetic':
        metadata = builder.generate_synthetic_dataset(
            episodes=args.episodes,
            policy=args.policy,
            output_path=output_path,
            seed=seed
        )
    else:  # semi-synthetic
        traffic_patterns = ['rush_hour', 'off_peak', 'mixed']
        metadata = builder.generate_semi_synthetic_dataset(
            episodes=args.episodes,
            traffic_patterns=traffic_patterns,
            output_path=output_path,
            seed=seed
        )
    
    logger.info(f"Dataset generation completed: {output_path}")
    logger.info(f"Dataset statistics: {metadata.get('statistics', {})}")
    print(f"\n✅ Dataset saved to: {output_path}")


def train_command(args):
    """Execute training command."""
    logger = get_logger("cli.train")
    logger.info(f"Starting training with config: {args.config}")
    
    # Validate config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Override experiment name if provided
    if args.experiment_name:
        # Load config, modify, and save temporarily
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['name'] = args.experiment_name
        
        # Save temporary config
        temp_config_path = f"{args.config}.temp"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        config_path = temp_config_path
    else:
        config_path = args.config
    
    try:
        # Initialize trainer
        # Determine device override
        device_override = None if args.device == 'auto' else args.device
        trainer = OfflineTrainer(config_path, device_override=device_override)
        
        # Resume from checkpoint if provided
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed training from: {args.resume}")
        
        # Run training
        results = trainer.train()
        
        logger.info("Training completed successfully")
        logger.info(f"Final results: {results.get('final_metrics', {})}")
        print(f"\n✅ Training completed for experiment: {results['experiment_name']}")
        
    finally:
        # Clean up temporary config
        if args.experiment_name and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def eval_command(args):
    """Execute evaluation command."""
    logger = get_logger("cli.eval")
    logger.info(f"Starting evaluation with config: {args.config}")
    
    # Validate config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Initialize evaluation runner with device override
    device_override = None if args.device == 'auto' else args.device
    evaluator = EvaluationRunner(args.config, args.checkpoint, device_override=device_override)
    
    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    logger.info("Evaluation completed")
    logger.info(f"Primary metric (Stopped Vehicle Ratio): {results['summary']['primary_metric']:.4f}")
    
    # Run baseline comparison if requested
    comparison_results = None
    if args.baseline_comparison:
        logger.info(f"Running baseline comparison: {args.baseline_comparison}")
        comparison_results = evaluator.compare_with_baseline(
            baseline_policy=args.baseline_comparison,
            num_episodes=min(args.episodes, 50)  # Limit baseline episodes
        )
        
        improvements = comparison_results.get('improvements', {})
        logger.info(f"Improvements vs {args.baseline_comparison}: {improvements}")
    
    # Save results if output file specified
    if args.output_file:
        import json
        output_data = {
            'evaluation_results': results,
            'comparison_results': comparison_results,
            'evaluation_config': {
                'episodes': args.episodes,
                'deterministic': args.deterministic,
                'baseline_comparison': args.baseline_comparison
            }
        }
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        # Convert numpy types to Python native before dumping
        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_native(v) for v in obj]
            try:
                import numpy as _np
                if isinstance(obj, (_np.generic,)):
                    return obj.item()
            except Exception:
                pass
            return obj

        native_output = _to_native(output_data)
        with open(args.output_file, 'w') as f:
            json.dump(native_output, f, indent=2)
        
        logger.info(f"Results saved to: {args.output_file}")
    
    # Print summary
    print(f"\n✅ Evaluation completed ({args.episodes} episodes)")
    print(f"Primary Metric (Stopped Vehicle Ratio): {results['summary']['primary_metric']:.4f}")
    print("Supporting Metrics:")
    for metric, value in results['summary']['supporting_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    if comparison_results:
        print(f"\nImprovements vs {args.baseline_comparison}:")
        for metric, improvement in comparison_results['improvements'].items():
            print(f"  {metric}: {improvement:+.1f}%")


def visualize_command(args):
    """Execute visualization command."""
    logger = get_logger("cli.visualize")
    logger.info(f"Creating visualizations for experiment: {args.experiment}")
    
    experiment_dir = os.path.join('experiments', args.experiment)
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(experiment_dir, 'plots')
    
    # Initialize plotter
    plotter = MARLOPlotter(output_dir=output_dir)
    
    # Load training history
    training_log_path = os.path.join(experiment_dir, 'logs', 'training_log.jsonl')
    training_history = load_training_history(training_log_path)
    
    # Look for evaluation results
    eval_results_path = os.path.join(experiment_dir, 'eval_results.json')
    evaluation_results = load_evaluation_results(eval_results_path) or {}
    
    # Look for comparison results
    comparison_results_path = os.path.join(experiment_dir, 'comparison_results.json')
    comparison_results = load_comparison_results(comparison_results_path) or {}
    
    plots_created = []
    
    # Generate requested plots
    if 'dashboard' in args.plots:
        # Create full dashboard
        dashboard_dir = plotter.create_dashboard(
            training_history, evaluation_results, comparison_results
        )
        plots_created.append(f"Dashboard in {dashboard_dir}")
        
    else:
        # Generate individual plots
        if 'learning' in args.plots and training_history:
            path = plotter.plot_learning_curve(training_history)
            if path:
                plots_created.append(f"Learning curve: {path}")
        
        if 'metrics' in args.plots and evaluation_results:
            path = plotter.plot_metrics_progression(evaluation_results)
            if path:
                plots_created.append(f"Metrics progression: {path}")
        
        if 'turns' in args.plots and evaluation_results:
            if 'action_analysis' in evaluation_results:
                path = plotter.plot_turn_distribution(evaluation_results['action_analysis'])
                if path:
                    plots_created.append(f"Turn distribution: {path}")
        
        if 'heatmap' in args.plots and evaluation_results:
            path = plotter.plot_queue_heatmap(evaluation_results)
            if path:
                plots_created.append(f"Queue heatmap: {path}")
        
        if 'comparison' in args.plots and comparison_results:
            # Try to make comparison heatmap if detailed results are present
            base = comparison_results.get('baseline_eval_results') or {}
            trained = comparison_results.get('trained_eval_results') or {}
            ch = plotter.plot_queue_heatmap_comparison(base, trained)
            if ch:
                plots_created.append(f"Queue heatmap comparison: {ch}")
        
        if 'comparison' in args.plots and comparison_results:
            path = plotter.plot_comparison_chart(comparison_results)
            if path:
                plots_created.append(f"Comparison chart: {path}")
    
    logger.info(f"Generated {len(plots_created)} visualizations")
    print(f"\n✅ Visualizations created:")
    for plot in plots_created:
        print(f"  {plot}")


def load_training_history(log_path: str) -> Dict[str, Any]:
    """Load training history from log file."""
    if not os.path.exists(log_path):
        return {}
    
    training_history = {}
    
    try:
        import json
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    log_entry = json.loads(line)
                    epoch = log_entry.pop('epoch', 0)
                    
                    for key, value in log_entry.items():
                        if key not in training_history:
                            training_history[key] = []
                        training_history[key].append(value)
        
    except Exception as e:
        print(f"Warning: Could not load training history: {e}")
    
    return training_history


def load_evaluation_results(results_path: str) -> Optional[Dict[str, Any]]:
    """Load evaluation results from file."""
    if not os.path.exists(results_path):
        return None
    
    try:
        import json
        with open(results_path, 'r') as f:
            data = json.load(f)
            payload = data.get('evaluation_results', data)
            # Coerce numeric strings to numbers
            def _coerce(obj):
                if isinstance(obj, dict):
                    return {k: _coerce(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_coerce(v) for v in obj]
                if isinstance(obj, str):
                    try:
                        if '.' in obj or 'e' in obj.lower():
                            return float(obj)
                        return int(obj)
                    except Exception:
                        return obj
                return obj
            return _coerce(payload)
    except Exception as e:
        print(f"Warning: Could not load evaluation results: {e}")
        return None


def load_comparison_results(results_path: str) -> Optional[Dict[str, Any]]:
    """Load comparison results from file."""
    if not os.path.exists(results_path):
        return None
    
    try:
        import json
        with open(results_path, 'r') as f:
            data = json.load(f)
            payload = data.get('comparison_results', data)
            # Coerce numeric strings to numbers
            def _coerce(obj):
                if isinstance(obj, dict):
                    return {k: _coerce(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_coerce(v) for v in obj]
                if isinstance(obj, str):
                    try:
                        if '.' in obj or 'e' in obj.lower():
                            return float(obj)
                        return int(obj)
                    except Exception:
                        return obj
                return obj
            return _coerce(payload)
    except Exception as e:
        print(f"Warning: Could not load comparison results: {e}")
        return None


if __name__ == "__main__":
    main()
