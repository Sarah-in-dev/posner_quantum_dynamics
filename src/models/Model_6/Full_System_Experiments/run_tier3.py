#!/usr/bin/env python3
"""
Run Tier 3 Experiments (Complete Suite)
========================================

Full-scale experiments testing complete quantum-classical cascade predictions.

These experiments run in minutes to hours and validate:

ORIGINAL EXPERIMENTS (validated):
1. Isotope comparison (P31 vs P32) - definitive test of quantum coherence necessity
2. Dopamine timing - eligibility decay validates T2 prediction
3. Pharmacological dissection - separate Q1/Q2/classical contributions
4. MT invasion - tryptophan network requirement

NEW EXPERIMENTS (December 2025):
5. Temperature - coherence vs kinetics tradeoff
6. Stimulation intensity - input-output curves (Hill analysis)
7. UV wavelength - optimal superradiance activation
8. Spatial clustering - geometry effects on cooperativity
9. Consolidation kinetics - classical cascade to structural plasticity

Known constraints (from Tier 2 debugging):
- O(n²) entanglement bottleneck: limit synapses to ~10, consolidation to ~1-2s
- Dopamine diffusion disabled: spatial gradient experiments deferred
- Gate timing sensitivity: factors must overlap temporally
- Use synapse.get_eligibility() not synapse.eligibility.get_eligibility()
- To block dopamine: params.dopamine = None
- To block calcium: params.calcium.nmda_blocked = True

Usage:
    python run_tier3.py                              # Full run, all experiments
    python run_tier3.py --quick                      # Quick validation mode
    python run_tier3.py --experiment isotope         # Single experiment
    python run_tier3.py --experiment temperature     # New experiment
    python run_tier3.py --list                       # List available experiments

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

# Import experiment modules - Original
from tier3_full_experiments import exp_isotope
from tier3_full_experiments import exp_dopamine_timing
from tier3_full_experiments import exp_pharmacology
from tier3_full_experiments import exp_mt_invasion

# Import experiment modules - New
from tier3_full_experiments import exp_temperature
from tier3_full_experiments import exp_stim_intensity
from tier3_full_experiments import exp_uv_wavelength
from tier3_full_experiments import exp_spatial_clustering
from tier3_full_experiments import exp_consolidation_kinetics
from tier3_full_experiments import exp_sensitivity_analysis
from tier3_full_experiments import exp_classical_comparison
from src.models.Model_6.Full_System_Experiments.tier5_rnn import exp_network_communication


EXPERIMENTS = {
    # === ORIGINAL EXPERIMENTS ===
    'isotope': {
        'name': 'Isotope Comparison (P31 vs P32)',
        'module': exp_isotope,
        'class': 'IsotopeExperiment',
        'description': 'Definitive test: P31 (T2≈100s) vs P32 (T2≈0.3s) eligibility decay',
        'category': 'core'
    },
    'dopamine_timing': {
        'name': 'Dopamine Timing Window',
        'module': exp_dopamine_timing,
        'class': 'DopamineTimingExperiment',
        'description': 'Validates T2=100s prediction through eligibility decay with delay',
        'category': 'core'
    },
    'pharmacology': {
        'name': 'Pharmacological Dissection',
        'module': exp_pharmacology,
        'class': 'PharmacologyExperiment',
        'description': 'Separate Q1 (tryptophan), Q2 (dimer), classical contributions',
        'category': 'core'
    },
    'mt_invasion': {
        'name': 'MT Invasion Requirement',
        'module': exp_mt_invasion,
        'class': 'MTInvasionExperiment',
        'description': 'Compare MT+ (invaded) vs MT- (naive) synapses',
        'category': 'core'
    },
    
    # === NEW EXPERIMENTS ===
    'temperature': {
        'name': 'Temperature Effects',
        'module': exp_temperature,
        'class': 'TemperatureExperiment',
        'description': 'Coherence vs kinetics tradeoff - validates physiological optimization',
        'category': 'physics'
    },
    'stim_intensity': {
        'name': 'Stimulation Intensity (IO Curves)',
        'module': exp_stim_intensity,
        'class': 'StimIntensityExperiment',
        'description': 'Input-output curves with Hill coefficient for cooperativity',
        'category': 'physics'
    },
    'uv_wavelength': {
        'name': 'UV Wavelength Effects',
        'module': exp_uv_wavelength,
        'class': 'UVWavelengthExperiment',
        'description': 'Optimal superradiance activation - validates tryptophan as Q1 substrate',
        'category': 'physics'
    },
    'spatial_clustering': {
        'name': 'Spatial Clustering',
        'module': exp_spatial_clustering,
        'class': 'SpatialClusteringExperiment',
        'description': 'Geometry effects on quantum cooperativity - clustered vs distributed',
        'category': 'architecture'
    },
    'consolidation': {
        'name': 'Consolidation Kinetics',
        'module': exp_consolidation_kinetics,
        'class': 'ConsolidationKineticsExperiment',
        'description': 'Full cascade to structural plasticity - validates quantum→classical coupling',
        'category': 'plasticity'
    },
    'sensitivity': {
    'name': 'Parameter Sensitivity Analysis',
    'module': exp_sensitivity_analysis,
    'class': 'SensitivityExperiment',
    'description': 'Tests robustness of predictions to parameter uncertainty',
    'category': 'validation'
    },
    'classical': {
    'name': 'Classical vs Quantum Comparison',
    'module': exp_classical_comparison,
    'class': 'ClassicalComparisonExperiment',
    'description': 'Shows CaMKII cannot bridge 60-100s learning window',
    'category': 'validation'
    },
    'network_communication': {
    'name': 'Network Communication via Entanglement (Gap 4)',
    'module': exp_network_communication,
    'class': 'NetworkCommunicationExperiment',
    'description': 'Tests coordination advantage: entangled O(N²) vs independent/classical O(2^N)',
    'category': 'core'
    },
    
}


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"tier3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_experiment(name: str, output_dir: Path,
                   quick: bool = False, verbose: bool = True) -> dict:
    """Run single experiment and return summary"""
    
    info = EXPERIMENTS[name]
    module = info['module']
    class_name = info['class']
    
    print(f"\n{'='*70}")
    print(f"TIER 3 EXPERIMENT: {info['name']}")
    print(f"{'='*70}")
    print(f"Category: {info['category']}")
    print(f"Description: {info['description']}")
    print()
    
    # === ADD THIS: Create experiment-specific subdirectory ===
    exp_output_dir = output_dir / name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the experiment class from the module
    ExperimentClass = getattr(module, class_name)
    experiment = ExperimentClass(quick_mode=quick, verbose=verbose)
    
    # Run experiment
    result = experiment.run()
    
    # Print summary
    experiment.print_summary(result)
    
    # Generate plots - USE exp_output_dir
    fig = experiment.plot(result, output_dir=exp_output_dir)
    
    # Save results - USE exp_output_dir
    result_path = exp_output_dir / f"results.json"  # Simplified name since it's in its own folder
    experiment.save_results(result, result_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run Tier 3 full-scale experiments (complete suite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tier3.py                           # Full run, all experiments
    python run_tier3.py --quick                   # Quick validation mode
    python run_tier3.py --experiment isotope      # Single experiment
    python run_tier3.py --experiment temperature  # New physics experiment
    python run_tier3.py --category core           # Run only core experiments
    python run_tier3.py --list                    # List available experiments
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer conditions, shorter durations)')
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Run single experiment')
    parser.add_argument('--category', type=str, 
                        choices=['core', 'physics', 'architecture', 'plasticity', 'all'],
                        default='all',
                        help='Run experiments by category')
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
        print("=" * 70)
        
        categories = {}
        for key, info in EXPERIMENTS.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((key, info))
        
        for cat in ['core', 'physics', 'architecture', 'plasticity']:
            if cat in categories:
                print(f"\n[{cat.upper()}]")
                print("-" * 60)
                for key, info in categories[cat]:
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
    elif args.category != 'all':
        experiments_to_run = [k for k, v in EXPERIMENTS.items() if v['category'] == args.category]
    else:
        experiments_to_run = list(EXPERIMENTS.keys())
    
    print(f"\nExperiments to run: {len(experiments_to_run)}")
    for exp in experiments_to_run:
        print(f"  - {exp}")
    
    # Run experiments
    for exp_name in experiments_to_run:
        try:
            result = run_experiment(
                exp_name, output_dir,
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
    
    # Key findings by category
    print("\n--- CORE EXPERIMENTS ---")
    for exp_name in ['isotope', 'dopamine_timing', 'pharmacology', 'mt_invasion']:
        if exp_name in all_results:
            result = all_results[exp_name]
            if result is None:
                print(f"  ? {exp_name}: No result returned")
            elif 'error' in result:
                print(f"  ✗ {exp_name}: ERROR - {result['error']}")
            elif exp_name == 'isotope' and result.get('quantum_necessary'):
                print(f"  ✓ ISOTOPE: Quantum coherence required (T2 ratio: {result.get('t2_ratio', 0):.0f}×)")
            elif exp_name == 'dopamine_timing':
                print(f"  ✓ DOPAMINE: Fitted T2 = {result.get('fitted_T2_s', 0):.1f}s")
            elif exp_name == 'pharmacology' and result.get('dissection_valid'):
                print(f"  ✓ PHARMACOLOGY: Q1/Q2/classical separated")
            elif exp_name == 'mt_invasion' and result.get('mt_required'):
                print(f"  ✓ MT INVASION: Field ratio = {result.get('field_ratio', 0):.1f}×")
    
    print("\n--- PHYSICS EXPERIMENTS ---")
    for exp_name in ['temperature', 'stim_intensity', 'uv_wavelength']:
        if exp_name in all_results:
            result = all_results[exp_name]
            if result is None:
                print(f"  ? {exp_name}: No result returned")
            elif 'error' in result:
                print(f"  ✗ {exp_name}: ERROR - {result['error']}")
            elif exp_name == 'temperature' and result.get('physiological_optimal'):
                print(f"  ✓ TEMPERATURE: Optimal at {result.get('optimal_temp_C', 37)}°C (physiological)")
            elif exp_name == 'stim_intensity' and result.get('cooperative'):
                print(f"  ✓ STIM INTENSITY: Hill n = {result.get('hill_n', 0):.2f} (cooperative)")
            elif exp_name == 'uv_wavelength' and result.get('matches_tryptophan'):
                print(f"  ✓ UV WAVELENGTH: Peak at {result.get('peak_wavelength_nm', 0):.0f}nm (matches Trp)")
    
    print("\n--- ARCHITECTURE EXPERIMENTS ---")
    for exp_name in ['spatial_clustering']:
        if exp_name in all_results:
            result = all_results[exp_name]
            if result is None:
                print(f"  ? {exp_name}: No result returned")
            elif 'error' in result:
                print(f"  ✗ {exp_name}: ERROR - {result['error']}")
            elif result.get('clustered_optimal'):
                print(f"  ✓ SPATIAL: Clustered geometry optimal")
    
    print("\n--- PLASTICITY EXPERIMENTS ---")
    for exp_name in ['consolidation']:
        if exp_name in all_results:
            result = all_results[exp_name]
            if result is None:
                print(f"  ? {exp_name}: No result returned")
            elif 'error' in result:
                print(f"  ✗ {exp_name}: ERROR - {result['error']}")
            elif result.get('feedback_validated'):
                print(f"  ✓ CONSOLIDATION: Cascade validated")
    
    if not args.no_plots:
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())