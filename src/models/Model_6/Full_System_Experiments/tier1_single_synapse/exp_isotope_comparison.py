"""
Tier 1 Experiment: Isotope Comparison (P31 vs P32)
===================================================

THE KILLER EXPERIMENT: Demonstrates that quantum coherence is functionally
necessary for learning, not just present as a side effect.

Physics:
- P31 (spin-1/2): T_singlet ~ 216s, dipolar relaxation only
- P32 (spin-1): T_singlet ~ 0.4s, quadrupolar relaxation dominates

Prediction:
- P31: Entanglement network persists through 60s learning window
- P32: Network collapses within seconds, no learning possible

This is the definitive test of the quantum cognition hypothesis.

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


@dataclass
class IsotopeResult:
    """Results from isotope comparison experiment"""
    isotope: str
    times: List[float] = field(default_factory=list)
    n_particles: List[int] = field(default_factory=list)
    n_bonds: List[int] = field(default_factory=list)
    largest_cluster: List[int] = field(default_factory=list)
    mean_singlet_prob: List[float] = field(default_factory=list)
    f_entangled: List[float] = field(default_factory=list)
    
    # Summary metrics
    final_particles: int = 0
    final_bonds: int = 0
    final_cluster: int = 0
    final_singlet: float = 0.0
    time_to_collapse: Optional[float] = None  # When P_S drops below 0.5


def run_single_isotope(fraction_P31: float, 
                       duration_s: float = 60.0,
                       verbose: bool = True) -> IsotopeResult:
    """
    Run experiment for single isotope condition
    
    Parameters
    ----------
    fraction_P31 : float
        1.0 for pure P31, 0.0 for pure P32
    duration_s : float
        Total duration of learning window
    verbose : bool
        Print progress
    """
    isotope = "P31" if fraction_P31 > 0.5 else "P32"
    result = IsotopeResult(isotope=isotope)
    
    # Configure model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = fraction_P31
    params.environment.fraction_P32 = 1.0 - fraction_P31
    
    model = Model6QuantumSynapse(params)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {isotope} (fraction_P31 = {fraction_P31})")
        print(f"{'='*60}")
    
    # === PHASE 1: THETA-BURST STIMULATION ===
    # 5 bursts at 5 Hz, each burst = 4 spikes at 100 Hz
    if verbose:
        print("Phase 1: Theta-burst stimulation")
    
    dt = 0.001  # 1 ms
    
    for burst in range(5):
        # Each burst: 4 spikes at 100 Hz (10ms intervals)
        for spike in range(4):
            # 2ms depolarization
            for _ in range(2):
                model.step(dt, {'voltage': -10e-3, 'reward': False})
            # 8ms at rest (within burst)
            for _ in range(8):
                model.step(dt, {'voltage': -70e-3, 'reward': False})
        
        # 160ms between bursts (to make 5 Hz)
        for _ in range(160):
            model.step(dt, {'voltage': -70e-3, 'reward': False})
        
        # Record after each burst
        pm = model.dimer_particles.get_network_metrics()
        if verbose:
            print(f"  Burst {burst+1}: particles={pm['n_dimers']}, "
                  f"bonds={pm['n_bonds']}, cluster={pm['largest_cluster']}")
    
    # === PHASE 2: LEARNING WINDOW ===
    if verbose:
        print(f"Phase 2: Learning window ({duration_s}s)")
    
    # Use coarser timestep for long evolution
    dt_rest = 0.1  # 100 ms steps for efficiency
    n_steps = int(duration_s / dt_rest)
    
    # Record at these timepoints
    record_times = [0, 1, 2, 5, 10, 15, 20, 30, 45, 60]
    
    t = 0.0
    collapse_detected = False
    
    for step in range(n_steps):
        model.step(dt_rest, {'voltage': -70e-3, 'reward': False})
        t += dt_rest
        
        # Record at specified times
        if any(abs(t - rt) < dt_rest/2 for rt in record_times) or step == n_steps - 1:
            pm = model.dimer_particles.get_network_metrics()
            
            # Calculate mean singlet probability
            if model.dimer_particles.dimers:
                mean_ps = np.mean([d.singlet_probability 
                                   for d in model.dimer_particles.dimers])
            else:
                mean_ps = 0.25  # Thermal if no dimers
            
            result.times.append(t)
            result.n_particles.append(pm['n_dimers'])
            result.n_bonds.append(pm['n_bonds'])
            result.largest_cluster.append(pm['largest_cluster'])
            result.mean_singlet_prob.append(mean_ps)
            result.f_entangled.append(pm['f_entangled'])
            
            # Detect collapse (P_S < 0.5)
            if mean_ps < 0.5 and not collapse_detected:
                result.time_to_collapse = t
                collapse_detected = True
            
            if verbose:
                status = "COLLAPSED" if mean_ps < 0.5 else "ENTANGLED"
                print(f"  t={t:5.1f}s: particles={pm['n_dimers']:2d}, "
                      f"bonds={pm['n_bonds']:2d}, cluster={pm['largest_cluster']:2d}, "
                      f"P_S={mean_ps:.3f} [{status}]")
    
    # Final metrics
    result.final_particles = result.n_particles[-1]
    result.final_bonds = result.n_bonds[-1]
    result.final_cluster = result.largest_cluster[-1]
    result.final_singlet = result.mean_singlet_prob[-1]
    
    return result


def run(duration_s: float = 60.0, verbose: bool = True) -> Dict[str, IsotopeResult]:
    """
    Run complete isotope comparison experiment
    
    Returns dict with 'P31' and 'P32' results
    """
    results = {}
    
    # Run P31
    results['P31'] = run_single_isotope(1.0, duration_s, verbose)
    
    # Run P32
    results['P32'] = run_single_isotope(0.0, duration_s, verbose)
    
    return results


def plot(results: Dict[str, IsotopeResult], output_dir: Path = None) -> plt.Figure:
    """
    Generate publication-quality figure
    
    Layout:
    - Top row: Singlet probability over time (main result)
    - Bottom left: Network size (particles, bonds)
    - Bottom right: Summary bar chart
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], 
                  hspace=0.3, wspace=0.3)
    
    # Colors
    color_P31 = '#2E86AB'  # Blue
    color_P32 = '#E94F37'  # Red
    
    p31 = results['P31']
    p32 = results['P32']
    
    # === TOP: SINGLET PROBABILITY (Main Result) ===
    ax_main = fig.add_subplot(gs[0, :])
    
    ax_main.plot(p31.times, p31.mean_singlet_prob, 'o-', 
                 color=color_P31, linewidth=2, markersize=6, label='³¹P (natural)')
    ax_main.plot(p32.times, p32.mean_singlet_prob, 's-', 
                 color=color_P32, linewidth=2, markersize=6, label='³²P (radioactive)')
    
    # Entanglement threshold
    ax_main.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, 
                    label='Entanglement threshold')
    ax_main.axhline(y=0.25, color='gray', linestyle=':', linewidth=1, 
                    label='Thermal equilibrium')
    
    # Shade regions
    ax_main.fill_between([0, max(p31.times)], 0.5, 1.0, alpha=0.1, color='green')
    ax_main.fill_between([0, max(p31.times)], 0.25, 0.5, alpha=0.1, color='red')
    
    ax_main.set_xlabel('Time (s)', fontsize=12)
    ax_main.set_ylabel('Mean Singlet Probability (P_S)', fontsize=12)
    ax_main.set_title('Isotope Comparison: Quantum Coherence During Learning Window', 
                      fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.set_xlim(0, max(p31.times) * 1.02)
    ax_main.set_ylim(0.2, 1.05)
    ax_main.grid(True, alpha=0.3)
    
    # Annotate key finding
    ax_main.annotate('P31: Network persists\n(learning possible)', 
                     xy=(45, 0.75), fontsize=10, color=color_P31,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_main.annotate('P32: Network collapsed\n(no learning)', 
                     xy=(10, 0.35), fontsize=10, color=color_P32,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === BOTTOM LEFT: NETWORK METRICS ===
    ax_network = fig.add_subplot(gs[1, 0])
    
    ax_network.plot(p31.times, p31.largest_cluster, 'o-', 
                    color=color_P31, linewidth=2, markersize=5, label='³¹P cluster')
    ax_network.plot(p32.times, p32.largest_cluster, 's-', 
                    color=color_P32, linewidth=2, markersize=5, label='³²P cluster')
    
    ax_network.set_xlabel('Time (s)', fontsize=11)
    ax_network.set_ylabel('Largest Entangled Cluster', fontsize=11)
    ax_network.set_title('Entanglement Network Size', fontsize=12)
    ax_network.legend(loc='upper right', fontsize=9)
    ax_network.set_xlim(0, max(p31.times) * 1.02)
    ax_network.grid(True, alpha=0.3)
    
    # === BOTTOM RIGHT: SUMMARY COMPARISON ===
    ax_summary = fig.add_subplot(gs[1, 1])
    
    metrics = ['Final\nCluster', 'Final\nBonds', 'P_S at\n60s']
    p31_values = [p31.final_cluster, p31.final_bonds, p31.final_singlet]
    p32_values = [p32.final_cluster, p32.final_bonds, p32.final_singlet]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax_summary.bar(x - width/2, p31_values, width, label='³¹P', color=color_P31)
    bars2 = ax_summary.bar(x + width/2, p32_values, width, label='³²P', color=color_P32)
    
    ax_summary.set_ylabel('Value', fontsize=11)
    ax_summary.set_title('Final State Comparison', fontsize=12)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(metrics, fontsize=10)
    ax_summary.legend(loc='upper right', fontsize=9)
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, p31_values):
        height = bar.get_height()
        ax_summary.annotate(f'{val:.2f}' if isinstance(val, float) else f'{val}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, p32_values):
        height = bar.get_height()
        ax_summary.annotate(f'{val:.2f}' if isinstance(val, float) else f'{val}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / 'isotope_comparison.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Also save data as JSON
        data = {
            'experiment': 'isotope_comparison',
            'timestamp': datetime.now().isoformat(),
            'P31': {
                'times': p31.times,
                'mean_singlet_prob': p31.mean_singlet_prob,
                'largest_cluster': p31.largest_cluster,
                'n_bonds': p31.n_bonds,
                'n_particles': p31.n_particles,
                'final_singlet': p31.final_singlet,
                'time_to_collapse': p31.time_to_collapse
            },
            'P32': {
                'times': p32.times,
                'mean_singlet_prob': p32.mean_singlet_prob,
                'largest_cluster': p32.largest_cluster,
                'n_bonds': p32.n_bonds,
                'n_particles': p32.n_particles,
                'final_singlet': p32.final_singlet,
                'time_to_collapse': p32.time_to_collapse
            },
            'conclusion': {
                'P31_maintains_entanglement': bool(p31.final_singlet > 0.5),
                'P32_collapses': bool(p32.final_singlet < 0.5),
                'quantum_necessary': bool(p31.final_singlet > 0.5 and p32.final_singlet < 0.5)
            }
        }
        
        json_path = output_dir / 'isotope_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {json_path}")
    
    return fig


def print_summary(results: Dict[str, IsotopeResult]):
    """Print summary of results"""
    print("\n" + "="*70)
    print("ISOTOPE COMPARISON SUMMARY")
    print("="*70)
    
    p31 = results['P31']
    p32 = results['P32']
    
    print(f"\n{'Metric':<25} {'³¹P':>15} {'³²P':>15} {'Ratio':>12}")
    print("-"*70)
    print(f"{'Final particles':<25} {p31.final_particles:>15} {p32.final_particles:>15}")
    print(f"{'Final bonds':<25} {p31.final_bonds:>15} {p32.final_bonds:>15}")
    print(f"{'Largest cluster':<25} {p31.final_cluster:>15} {p32.final_cluster:>15}")
    print(f"{'Final P_S':<25} {p31.final_singlet:>15.3f} {p32.final_singlet:>15.3f}")
    
    if p32.time_to_collapse:
        print(f"{'Time to collapse':<25} {'N/A':>15} {p32.time_to_collapse:>14.1f}s")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if p31.final_singlet > 0.5 and p32.final_singlet < 0.5:
        print("""
✓ P31 maintains entanglement through 60s learning window
✗ P32 network collapses within seconds

This demonstrates that QUANTUM COHERENCE IS FUNCTIONALLY NECESSARY
for the learning mechanism - not just present as a side effect.

Experimental prediction: Animals fed P32-enriched phosphate will show
impaired synaptic plasticity and learning.
""")
    else:
        print("WARNING: Results do not match prediction - check model parameters")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run isotope comparison experiment')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Learning window duration in seconds')
    parser.add_argument('--output', type=str, default='results/tier1',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run(duration_s=args.duration, verbose=not args.quiet)
    
    # Print summary
    print_summary(results)
    
    # Generate figure
    fig = plot(results, output_dir=args.output)
    
    plt.show()