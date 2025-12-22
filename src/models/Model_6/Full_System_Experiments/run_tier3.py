#!/usr/bin/env python3
"""
Run Tier 3 Experiments
======================

Full-scale experiments testing complete quantum-classical cascade predictions.

These experiments run in minutes to hours and validate:
1. Isotope comparison (P31 vs P32) - definitive test of quantum coherence necessity
2. Dopamine timing - eligibility decay validates T2 prediction
3. Pharmacological dissection - separate Q1/Q2/classical contributions
4. MT invasion - tryptophan network requirement

Known constraints (from Tier 2 debugging):
- O(n²) entanglement bottleneck: limit synapses to ~10, consolidation to ~1-2s
- Dopamine diffusion disabled: spatial gradient experiments deferred
- Gate timing sensitivity: factors must overlap temporally
- Use synapse.get_eligibility() not synapse.eligibility.get_eligibility()
- To block dopamine: params.dopamine = None
- To block calcium: params.calcium.nmda_blocked = True

Usage:
    python run_tier3.py                           # Full run, all experiments
    python run_tier3.py --quick                   # Quick validation mode
    python run_tier3.py --experiment isotope      # Single experiment
    python run_tier3.py --list                    # List available experiments

Author: Sarah Davidson
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add parent to path (Model_6 directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import experiment modules
from tier3_full_experiments import exp_isotope
from tier3_full_experiments import exp_dopamine_timing
from tier3_full_experiments import exp_pharmacology
from tier3_full_experiments import exp_mt_invasion


EXPERIMENTS = {
    'isotope': {
        'name': 'Isotope Comparison (P31 vs P32)',
        'module': exp_isotope,
        'description': 'Definitive test: P31 (T2≈67s) vs P32 (T2≈0.3s) eligibility decay'
    },
    'dopamine_timing': {
        'name': 'Dopamine Timing Window',
        'module': exp_dopamine_timing,
        'description': 'Validates T2=67s prediction through eligibility decay with delay'
    },
    'pharmacology': {
        'name': 'Pharmacological Dissection',
        'module': exp_pharmacology,
        'description': 'Separate Q1 (tryptophan), Q2 (dimer), classical contributions'
    },
    'mt_invasion': {
        'name': 'MT Invasion Requirement',
        'module': exp_mt_invasion,
        'description': 'Compare MT+ (invaded) vs MT- (naive) synapses'
    },
}


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"tier3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_experiment(name: str, module, output_dir: Path,
                   quick: bool = False, verbose: bool = True) -> dict:
    """Run single experiment and return summary"""
    
    print(f"\n{'='*70}")
    print(f"TIER 3 EXPERIMENT: {EXPERIMENTS[name]['name']}")
    print(f"{'='*70}")
    print(EXPERIMENTS[name]['description'])
    print()
    
    # Get the experiment class from the module
    if name == 'isotope':
        experiment = module.IsotopeExperiment(quick_mode=quick, verbose=verbose)
    elif name == 'dopamine_timing':
        experiment = module.DopamineTimingExperiment(quick_mode=quick, verbose=verbose)
    elif name == 'pharmacology':
        experiment = module.PharmacologyExperiment(quick_mode=quick, verbose=verbose)
    elif name == 'mt_invasion':
        experiment = module.MTInvasionExperiment(quick_mode=quick, verbose=verbose)
    else:
        raise ValueError(f"Unknown experiment: {name}")
    
    # Run experiment
    result = experiment.run()
    
    # Print summary
    experiment.print_summary(result)
    
    # Generate plots
    fig = experiment.plot(result, output_dir=output_dir)
    
    # Save results
    result_path = output_dir / f"{name}_results.json"
    experiment.save_results(result, result_path)
    
    return experiment.get_summary_dict(result)


def main():
    parser = argparse.ArgumentParser(
        description='Run Tier 3 full-scale experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tier3.py                           # Full run, all experiments
    python run_tier3.py --quick                   # Quick validation mode
    python run_tier3.py --experiment isotope      # Single experiment
    python run_tier3.py --list                    # List available experiments
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer conditions, shorter durations)')
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Run single experiment')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot display (still saves figures)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Tier 3 Experiments:")
        print("-" * 60)
        for key, info in EXPERIMENTS.items():
            print(f"  {key:<20} - {info['name']}")
            print(f"  {' '*20}   {info['description']}")
        return 0
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"\nOutput directory: {output_dir}")
    
    # Track results
    all_results = {}
    
    # Determine which experiments to run
    if args.experiment:
        experiments_to_run = [args.experiment]
    else:
        experiments_to_run = list(EXPERIMENTS.keys())
    
    # Run experiments
    for exp_name in experiments_to_run:
        module = EXPERIMENTS[exp_name]['module']
        try:
            result = run_experiment(
                exp_name, module, output_dir,
                quick=args.quick,
                verbose=not args.quiet
            )
            all_results[exp_name] = result
        except Exception as e:
            print(f"\nERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}
    
    # Save master summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick' if args.quick else 'full',
        'experiments_run': experiments_to_run,
        'results': all_results
    }
    
    summary_path = output_dir / 'tier3_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("TIER 3 EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Experiments run: {len(experiments_to_run)}")
    
    # Key findings
    for exp_name, result in all_results.items():
        if 'error' in result:
            print(f"\n✗ {exp_name}: ERROR - {result['error']}")
        elif exp_name == 'isotope':
            if result.get('quantum_necessary'):
                print(f"\n✓ ISOTOPE: P31 maintains eligibility, P32 does not - quantum coherence required")
            else:
                print(f"\n⚠ ISOTOPE: Check results - expected P31 >> P32")
        elif exp_name == 'dopamine_timing':
            fitted_t2 = result.get('fitted_T2_s', 0)
            print(f"\n✓ DOPAMINE TIMING: Fitted T2 = {fitted_t2:.1f}s (theory: 67s)")
        elif exp_name == 'pharmacology':
            if result.get('dissection_valid'):
                print(f"\n✓ PHARMACOLOGY: Q1/Q2/classical contributions separated")
        elif exp_name == 'mt_invasion':
            if result.get('mt_required'):
                print(f"\n✓ MT INVASION: Tryptophan network required for full cascade")
    
    if not args.no_plots:
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())