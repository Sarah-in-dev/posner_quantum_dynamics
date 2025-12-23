#!/usr/bin/env python3
"""
Run Tier 4 Experiments: Feedback Loop Characterization
========================================================

Emergent/multi-cycle experiments testing bidirectional quantum-classical coupling.

Unlike Tier 1-3 experiments which validate individual components and one-pass
cascade behavior, Tier 4 tests EMERGENT BEHAVIORS that arise from the complete
system with feedback coupling:

1. POSITIVE FEEDBACK AMPLIFICATION
   - Tryptophan EM fields enhance dimer formation
   - Dimer fields enhance tryptophan coupling
   - Together: gain > 1 amplification

2. BISTABILITY
   - Two stable states (low/high activity)
   - System can lock into either state
   - Partial stimulation → bimodal outcomes

3. HYSTERESIS (future)
   - Different thresholds for switching up vs down
   - History-dependent behavior

4. MULTI-CYCLE SELF-ORGANIZATION
   - Each stimulation cycle potentiates the next
   - Progressive enhancement over repeated activity
   - Demonstrates memory-like behavior

Scientific basis:
The system contains TWO coupled feedback loops:

LOOP 1 - Fast (Forward Coupling):
  Tryptophan superradiance → EM field (20 kT) → Enhanced dimer formation
  Timescale: milliseconds to seconds

LOOP 2 - Slow (Reverse Coupling):
  Dimer coherent fields (30 kT) → Tubulin conformational modulation → Better Trp coupling
  Timescale: seconds to minutes

Usage:
    python run_tier4.py                    # Full run
    python run_tier4.py --quick            # Quick validation mode

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

# Import experiment module
from tier4_feedback import exp_feedback_loop


EXPERIMENTS = {
    'feedback_loop': {
        'name': 'Feedback Loop Characterization',
        'module': exp_feedback_loop,
        'class': 'FeedbackLoopExperiment',
        'description': 'Tests positive feedback, bistability, and multi-cycle enhancement',
    },
}


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"tier4_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_experiment(name: str, output_dir: Path,
                   quick: bool = False, verbose: bool = True) -> dict:
    """Run single experiment and return summary"""
    
    info = EXPERIMENTS[name]
    module = info['module']
    class_name = info['class']
    
    print("\n" + "="*70)
    print("TIER 4: EMERGENT BEHAVIOR / FEEDBACK LOOP EXPERIMENTS")
    print("="*70)
    print(f"\nExperiment: {info['name']}")
    print(f"Description: {info['description']}")
    print()
    
    # Get the experiment class from the module
    ExperimentClass = getattr(module, class_name)
    experiment = ExperimentClass(quick_mode=quick, verbose=verbose)
    
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
        description='Run Tier 4 feedback loop experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tier4.py                    # Full run
    python run_tier4.py --quick            # Quick validation mode

This tier tests emergent behaviors from bidirectional quantum-classical coupling:
- Positive feedback amplification
- Bistability (two stable states)
- Multi-cycle self-organization
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer trials, shorter durations)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot display (still saves figures)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    print(f"\nOutput directory: {output_dir}")
    
    # Track results
    all_results = {}
    
    # Run feedback loop experiment
    try:
        result = run_experiment(
            'feedback_loop', output_dir,
            quick=args.quick,
            verbose=not args.quiet
        )
        all_results['feedback_loop'] = result
    except Exception as e:
        print(f"\nERROR in feedback_loop: {e}")
        import traceback
        traceback.print_exc()
        all_results['feedback_loop'] = {'error': str(e)}
    
    # Save master summary
    summary = {
        'tier': 4,
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick' if args.quick else 'full',
        'results': all_results
    }
    
    summary_path = output_dir / 'tier4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("TIER 4 EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    
    # Key findings
    result = all_results.get('feedback_loop', {})
    
    if 'error' in result:
        print(f"\n✗ ERROR: {result['error']}")
    else:
        print("\n--- FEEDBACK LOOP ANALYSIS ---")
        
        amp = result.get('amplification_factor', 1)
        if amp > 1.5:
            print(f"  ✓ POSITIVE FEEDBACK: {amp:.2f}× amplification")
        else:
            print(f"  ~ Weak feedback: {amp:.2f}× amplification")
        
        if result.get('bistability_detected'):
            print(f"  ✓ BISTABILITY: Two stable states detected")
            print(f"    High state: {result.get('high_state_fraction', 0):.0%}")
            print(f"    Low state: {result.get('low_state_fraction', 0):.0%}")
        else:
            print("  ~ No clear bistability")
        
        multi = result.get('multicycle_enhancement', 1)
        if multi > 1.3:
            print(f"  ✓ MULTICYCLE ENHANCEMENT: {multi:.2f}× over cycles")
        else:
            print(f"  ~ Limited enhancement: {multi:.2f}× over cycles")
        
        if result.get('feedback_validated'):
            print("\n" + "="*70)
            print("✓ FEEDBACK LOOP VALIDATED")
            print("  The quantum-classical system demonstrates emergent self-organization")
            print("="*70)
        else:
            print("\n  ⚠ PARTIAL VALIDATION - some feedback effects present")
    
    if not args.no_plots:
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())