#!/usr/bin/env python3
"""
Run Tier 2 Experiments
======================

Executes minimal multi-synapse experiments testing network-level effects.

These experiments run in minutes and validate:
1. Network threshold for commitment (synapse count scaling)
2. Three-factor AND-gate (eligibility + dopamine + calcium required)

Usage:
    python run_tier2.py                    # Full run
    python run_tier2.py --quick            # Quick validation
    python run_tier2.py --experiment network  # Single experiment

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
sys.path.insert(0, str(Path(__file__).parent))

from tier2_minimal_network import exp_network_threshold
from tier2_minimal_network import exp_three_factor_gate


EXPERIMENTS = {
    'network_threshold': {
        'name': 'Network Threshold',
        'module': exp_network_threshold,
        'description': 'How synapse count affects commitment'
    },
    'three_factor_gate': {
        'name': 'Three-Factor Gate',
        'module': exp_three_factor_gate,
        'description': 'Verify eligibility + dopamine + calcium required'
    }
}


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"tier2_{timestamp}"
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
    
    if name == 'network_threshold':
        # Quick mode: fewer synapse counts
        if quick:
            synapse_counts = [1, 3, 5]
        else:
            synapse_counts = [1, 3, 5, 10]
        
        result = module.run(synapse_counts=synapse_counts, verbose=verbose)
        module.print_summary(result)
        fig = module.plot(result, output_dir=exp_output_dir)
        
        return {
            'threshold_n_synapses': result.threshold_n_synapses,
            'conditions_tested': len(result.conditions),
            'any_committed': any(c.committed for c in result.conditions.values())
        }
    
    elif name == 'three_factor_gate':
        result = module.run(verbose=verbose)
        module.print_summary(result)
        fig = module.plot(result, output_dir=exp_output_dir)
        
        return {
            'gate_validated': bool(result.gate_logic_validated),
            'control_committed': bool(result.conditions['control'].committed),
            'no_elig_committed': bool(result.conditions['no_eligibility'].committed),
            'no_dopa_committed': bool(result.conditions['no_dopamine'].committed),
            'no_ca_committed': bool(result.conditions['no_calcium'].committed)
        }
    
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run Tier 2 multi-synapse experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tier2.py                           # Full run
    python run_tier2.py --quick                   # Quick validation
    python run_tier2.py --experiment network_threshold  # Single experiment
    python run_tier2.py --list                    # List experiments
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer conditions)')
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Run single experiment')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot display')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Tier 2 Experiments:")
        print("-" * 50)
        for key, info in EXPERIMENTS.items():
            print(f"  {key:<20} - {info['name']}")
        return 0
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"\nOutput directory: {output_dir}")
    
    # Track results
    all_results = {}
    
    # Run experiments
    if args.experiment:
        experiments_to_run = [args.experiment]
    else:
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
    print("TIER 2 EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Experiments run: {len(experiments_to_run)}")
    
    # Key findings
    if 'three_factor_gate' in all_results:
        gate = all_results['three_factor_gate']
        if gate.get('gate_validated'):
            print("\n✓ THREE-FACTOR GATE VALIDATED: All factors required")
        else:
            print("\n⚠ Three-factor gate: Check results")
    
    if 'network_threshold' in all_results:
        net = all_results['network_threshold']
        if net.get('threshold_n_synapses'):
            print(f"✓ NETWORK THRESHOLD: {net['threshold_n_synapses']} synapses")
        elif net.get('any_committed'):
            print("✓ NETWORK: Some commitment observed")
        else:
            print("⚠ NETWORK: No commitment in tested range")
    
    if not args.no_plots:
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())