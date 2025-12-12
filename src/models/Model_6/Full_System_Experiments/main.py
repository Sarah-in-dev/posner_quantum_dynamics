#!/usr/bin/env python3
"""
Quantum Cascade Experimental Framework - Main Entry Point
===========================================================

Run the complete experimental suite characterizing the quantum-classical cascade.

Usage:
    # Quick test (2 trials, short consolidation)
    python main.py --quick
    
    # Full run (5 trials, 60s consolidation)
    python main.py --full
    
    # Custom configuration
    python main.py --workers 8 --trials 10 --output results_v2
    
    # Single experiment
    python main.py --experiment network_threshold

Author: Sarah Davidson
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import ExperimentConfig, print_section_header
from experiments import (
    run_all_experiments,
    experiment_network_threshold,
    experiment_mt_invasion,
    experiment_uv_wavelength,
    experiment_anesthetic,
    experiment_stim_intensity,
    experiment_dopamine_timing,
    experiment_temperature,
    experiment_pharmacology,
    experiment_isotope,
    experiment_spatial_clustering,
    experiment_consolidation,
    experiment_three_factor_gate,
)
from visualizations import generate_all_figures


# Map experiment names to functions
EXPERIMENTS = {
    'network_threshold': experiment_network_threshold,
    'mt_invasion': experiment_mt_invasion,
    'uv_wavelength': experiment_uv_wavelength,
    'anesthetic': experiment_anesthetic,
    'stim_intensity': experiment_stim_intensity,
    'dopamine_timing': experiment_dopamine_timing,
    'temperature': experiment_temperature,
    'pharmacology': experiment_pharmacology,
    'isotope': experiment_isotope,
    'spatial_clustering': experiment_spatial_clustering,
    'consolidation': experiment_consolidation,
    'three_factor_gate': experiment_three_factor_gate,
}


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Quantum Cascade Experimental Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --quick                    # Quick test run
    python main.py --full                     # Full production run
    python main.py --workers 8 --trials 10    # Custom config
    python main.py --experiment isotope       # Single experiment
    python main.py --list                     # List all experiments
        """
    )
    
    # Mode
    mode = parser.add_mutually_exclusive_group()
    
    mode.add_argument('--smoke', action='store_true', 
                  help='Ultra-fast smoke test (1 trial, short durations)')
    mode.add_argument('--quick', action='store_true',
                      help='Quick test mode (2 trials, short consolidation)')
    mode.add_argument('--full', action='store_true',
                      help='Full production mode (10 trials, 60s consolidation)')
    
    # Configuration
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Trials per condition (default: 5)')
    parser.add_argument('--consolidation', type=float, default=60.0,
                        help='Consolidation duration in seconds (default: 60)')
    
    # Output
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    
    # Experiment selection
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Run single experiment')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    
    # Parallelization
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel execution')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    args = parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 40)
        for name in EXPERIMENTS:
            print(f"  {name}")
        print("\nRun with: python main.py --experiment <name>")
        return 0
    
    # Build configuration
    config = ExperimentConfig()
    
    # Track which experiments to run (None = all)
    experiments_to_run = None
    
    if args.smoke:
        config.quick_mode = True
        config.n_trials = 1
        config.consolidation_duration = 1.0
        config.baseline_duration = 0.1
        config.n_workers = min(4, args.workers)
        # Only run critical experiments for smoke test
        experiments_to_run = ['isotope', 'three_factor_gate', 'network_threshold']
        print("\n🔥 SMOKE TEST MODE - minimal validation run")
    elif args.quick:
        config.quick_mode = True
        config.n_trials = 2
        config.consolidation_duration = 10.0
        config.n_workers = min(4, args.workers)
    elif args.full:
        config.quick_mode = False
        config.n_trials = 10
        config.consolidation_duration = 60.0
        config.n_workers = args.workers
    else:
        config.quick_mode = False
        config.n_trials = args.trials
        config.consolidation_duration = args.consolidation
        config.n_workers = args.workers
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"{args.output}/run_{timestamp}"
    config.use_multiprocessing = not args.no_parallel
    config.save_figures = not args.no_figures
    
    # Print header
    print_section_header("QUANTUM CASCADE EXPERIMENTAL FRAMEWORK")
    print(f"\nConfiguration:")
    print(f"  Mode: {'QUICK' if config.quick_mode else 'FULL'}")
    print(f"  Workers: {config.n_workers}")
    print(f"  Trials per condition: {config.n_trials}")
    print(f"  Consolidation: {config.consolidation_duration}s")
    print(f"  Output: {config.output_dir}/")
    print(f"  Parallel: {config.use_multiprocessing}")
    
    # Run experiments
    if args.experiment:
        # Single experiment
        print(f"\nRunning single experiment: {args.experiment}")
        exp_func = EXPERIMENTS[args.experiment]
        result = exp_func(config, verbose=True)
        results = {args.experiment: result}
    elif experiments_to_run:
        # Subset of experiments (smoke mode)
        print(f"\nRunning {len(experiments_to_run)} experiments: {experiments_to_run}")
        results = {}
        for exp_name in experiments_to_run:
            if exp_name in EXPERIMENTS:
                print(f"\n--- {exp_name} ---")
                exp_func = EXPERIMENTS[exp_name]
                results[exp_name] = exp_func(config, verbose=True)
            else:
                print(f"Warning: Unknown experiment '{exp_name}', skipping")
    else:
        # All experiments
        results = run_all_experiments(config, verbose=True)
    
    # Generate figures
    if config.save_figures and results:
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)
        generate_all_figures(results, config.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {config.output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())