#!/usr/bin/env python3
"""
Run Tier 1 Experiments
======================

Executes all single-synapse experiments demonstrating core quantum physics.

These experiments run in seconds-to-minutes and validate:
1. Isotope discrimination (P31 vs P32)
2. EM-mediated entanglement network formation
3. Theta-burst calcium/dimer integration
4. Singlet probability decay physics

Usage:
    python run_tier1.py                    # Full run
    python run_tier1.py --quick            # Quick validation
    python run_tier1.py --experiment isotope  # Single experiment
    python run_tier1.py --no-plots         # Skip visualization

Author: Sarah Davidson
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tier1_single_synapse import exp_isotope_comparison
from tier1_single_synapse import exp_entanglement_network
from tier1_single_synapse import exp_theta_burst
from tier1_single_synapse import exp_coherence_decay


EXPERIMENTS = {
    'isotope': {
        'name': 'Isotope Comparison (P31 vs P32)',
        'module': exp_isotope_comparison,
        'description': 'THE KILLER EXPERIMENT: Quantum coherence necessity'
    },
    'entanglement': {
        'name': 'Entanglement Network Formation',
        'module': exp_entanglement_network,
        'description': 'EM field as quantum bus for inter-dimer coupling'
    },
    'theta_burst': {
        'name': 'Theta-Burst Integration',
        'module': exp_theta_burst,
        'description': 'Multi-spike calcium/dimer accumulation'
    },
    'coherence_decay': {
        'name': 'Coherence Decay Validation',
        'module': exp_coherence_decay,
        'description': 'Singlet probability matches Agarwal physics'
    }
}


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"tier1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_experiment(name: str, module, output_dir: Path, 
                   quick: bool = False, verbose: bool = True) -> dict:
    """Run single experiment and return summary"""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {EXPERIMENTS[name]['name']}")
    print(f"{'='*70}")
    print(EXPERIMENTS[name]['description'])
    
    # Create experiment-specific subdirectory
    exp_output_dir = output_dir / name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure based on quick mode
    if name == 'isotope':
        duration = 10.0 if quick else 60.0
        results = module.run(duration_s=duration, verbose=verbose)
        module.print_summary(results)
        fig = module.plot(results, output_dir=exp_output_dir)
        
        return {
            'P31_final_singlet': float(results['P31'].final_singlet),
            'P32_final_singlet': float(results['P32'].final_singlet),
            'quantum_necessary': bool(results['P31'].final_singlet > 0.5 and results['P32'].final_singlet < 0.5)
        }
    
    elif name == 'entanglement':
        n_bursts = 3 if quick else 5
        rest = 3.0 if quick else 10.0
        result = module.run(n_bursts=n_bursts, rest_duration_s=rest, verbose=verbose)
        module.print_summary(result)
        fig = module.plot(result, output_dir=exp_output_dir)
        
        return {
            'final_particles': int(result.rest_particles[-1]) if result.rest_particles else 0,
            'final_bonds': int(result.rest_bonds[-1]) if result.rest_bonds else 0,
            'network_coverage': float(result.final_network_fraction)
        }
    
    elif name == 'theta_burst':
        n_bursts = 3 if quick else 5
        result = module.run(n_bursts=n_bursts, verbose=verbose)
        module.print_summary(result)
        fig = module.plot(result, output_dir=exp_output_dir)
        
        return {
            'final_particles': int(result.final_particles),
            'final_dimer_nM': float(result.final_dimer_conc_nM),
            'calcium_integral': float(result.total_calcium_integral)
        }
    
    elif name == 'coherence_decay':
        dur_p31 = 60.0 if quick else 300.0
        dur_p32 = 2.0 if quick else 5.0
        results = module.run(duration_P31=dur_p31, duration_P32=dur_p32, verbose=verbose)
        module.print_summary(results)
        fig = module.plot(results, output_dir=exp_output_dir)
        
        return {
            'P31_T_fitted': float(results['P31'].T_singlet_fitted),
            'P32_T_fitted': float(results['P32'].T_singlet_fitted),
            'P31_T_theory': float(results['P31'].T_singlet_theory),
            'P32_T_theory': float(results['P32'].T_singlet_theory)
        }
    
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run Tier 1 single-synapse experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tier1.py                      # Full run, all experiments
    python run_tier1.py --quick              # Quick validation mode
    python run_tier1.py --experiment isotope # Single experiment
    python run_tier1.py --list               # List available experiments
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (shorter durations)')
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
        print("\nAvailable Tier 1 Experiments:")
        print("-" * 50)
        for key, info in EXPERIMENTS.items():
            print(f"  {key:<15} - {info['name']}")
        return 0
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"\nOutput directory: {output_dir}")
    
    # Track results
    all_results = {}
    
    # Run experiments
    if args.experiment:
        # Single experiment
        experiments_to_run = [args.experiment]
    else:
        # All experiments
        experiments_to_run = list(EXPERIMENTS.keys())
    
    for exp_name in experiments_to_run:
        module = EXPERIMENTS[exp_name]['module']
        result = run_experiment(
            exp_name, module, output_dir,
            quick=args.quick,
            verbose=not args.quiet
        )
        all_results[exp_name] = result
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick' if args.quick else 'full',
        'experiments_run': experiments_to_run,
        'results': all_results
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("TIER 1 EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Experiments run: {len(experiments_to_run)}")
    
    # Key findings
    if 'isotope' in all_results:
        iso = all_results['isotope']
        if iso.get('quantum_necessary'):
            print("\n✓ ISOTOPE TEST PASSED: Quantum coherence is functionally necessary")
        else:
            print("\n✗ ISOTOPE TEST: Check results")
    
    if not args.no_plots:
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())