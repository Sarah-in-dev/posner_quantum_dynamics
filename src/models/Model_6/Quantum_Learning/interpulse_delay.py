"""
Interpulse Delay Experiment - Model 6 Integration
==================================================

Tests the quantum eligibility trace prediction using the full Model 6 synapse.

Protocol:
    1. Presynaptic burst (creates dimers, establishes eligibility)
    2. Variable delay (eligibility decays with T2)
    3. Plateau potential (converts eligibility → plasticity)
    4. Measure final synaptic strength

Key prediction: Half-decay delay should match T2
    - P31: ~47 seconds
    - P32: ~0.21 seconds
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from eligibility_trace import PhosphorusIsotope


def run_single_trial(delay_s: float, isotope: str = 'P31', verbose: bool = False) -> Dict:
    """
    Run single interpulse delay trial
    
    Args:
        delay_s: Delay between presynaptic burst and plateau (seconds)
        isotope: 'P31' or 'P32'
        verbose: Print progress
        
    Returns:
        Dict with trial results
    """
    # Initialize model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    model = Model6QuantumSynapse(params)
    
    # Set isotope
    if isotope == 'P32':
        model.quantum.P31_fraction = 0.0  # 100% P32
    else:
        model.quantum.P31_fraction = 1.0  # 100% P31
    
    dt = 0.001  # 1 ms timestep
    
    # Record baseline
    baseline_strength = model.spine_plasticity.get_synaptic_strength()
    
    # === PHASE 1: Presynaptic burst (30 ms) ===
    # High activity creates calcium spike → dimers → eligibility
    if verbose:
        print(f"  Phase 1: Presynaptic burst...")
    
    for _ in range(30):  # 30 ms
        stimulus = {
            'voltage': -10e-3,  # Depolarized
            'activity_level': 1.0,
            'plateau_potential': False
        }
        model.step(dt, stimulus)
    
    eligibility_after_burst = model.eligibility.get_eligibility()
    
    # === PHASE 2: Delay period ===
    # Resting state, eligibility decays
    if verbose:
        print(f"  Phase 2: Delay ({delay_s}s)...")
    
    n_delay_steps = int(delay_s / dt)
    for _ in range(n_delay_steps):
        stimulus = {
            'voltage': -70e-3,  # Resting
            'activity_level': 0.0,
            'plateau_potential': False
        }
        model.step(dt, stimulus)
    
    eligibility_before_plateau = model.get_eligibility()

    # Lock this eligibility value for the plateau phase
    model.set_locked_eligibility(eligibility_before_plateau)

    # === RESET CaMKII BEFORE PLATEAU ===
    # Isolates eligibility trace effect - only coherence differs between delays
    model.camkii.pT286 = 0.0
    model.camkii.CaCaM_bound = 0.0
    model.camkii.CaMKII_active = 0.0
    model.camkii.GluN2B_bound = 0.0
    model.camkii.molecular_memory = 0.0

    # Reset spine to baseline before plateau
    model.spine_plasticity.AMPAR_count = model.spine_plasticity.params.ampar.baseline_AMPAR
    model.spine_plasticity.AMPAR_surface = model.spine_plasticity.params.ampar.baseline_AMPAR
    model.spine_plasticity.spine_volume = 1.0
    model.spine_plasticity.actin_dynamic = model.spine_plasticity.params.actin.dynamic_pool_baseline
    model.spine_plasticity.actin_stable = model.spine_plasticity.params.actin.stable_pool_baseline


    # Measure strength right before plateau (new baseline for this measurement)
    pre_plateau_strength = model.spine_plasticity.get_synaptic_strength()
    
    # === PHASE 3: Plateau potential (300 ms) ===
    # Dendritic calcium spike - the instructive signal
    if verbose:
        print(f"  Phase 3: Plateau potential...")
    
    plasticity_triggered = False
    for _ in range(300):  # 300 ms
        stimulus = {
            'voltage': -20e-3,  # Plateau depolarization
            'activity_level': 0.5,
            'plateau_potential': True  # THIS IS THE KEY
        }
        model.step(dt, stimulus)
        
        if model._plasticity_gate:
            plasticity_triggered = True
    
    # === PHASE 4: Recovery (1 s) ===
    if verbose:
        print(f"  Phase 4: Recovery...")
    
    for _ in range(1000):
        stimulus = {
            'voltage': -70e-3,
            'activity_level': 0.0,
            'plateau_potential': False
        }
        model.step(dt, stimulus)
    
    final_strength = model.spine_plasticity.get_synaptic_strength()
    
    return {
        'delay_s': delay_s,
        'isotope': isotope,
        'baseline_strength': baseline_strength,
        'pre_plateau_strength': pre_plateau_strength,
        'final_strength': final_strength,
        'strength_change': final_strength - baseline_strength,
        'normalized_change': (final_strength - baseline_strength) / baseline_strength,
        'eligibility_after_burst': eligibility_after_burst,
        'eligibility_before_plateau': eligibility_before_plateau,
        'plasticity_triggered': plasticity_triggered,
        'T2': model.quantum.get_experimental_metrics()['T2_dimer_s'],
    }


def run_delay_sweep(isotope: str, delays: List[float], n_trials: int = 1, verbose: bool = True) -> List[Dict]:
    """
    Run sweep across multiple delays
    """
    results = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {isotope} delay sweep")
        print(f"T2 = {67.6 if isotope == 'P31' else 0.3} s")
        print(f"{'='*60}")
    
    for i, delay in enumerate(delays):
        for trial in range(n_trials):
            if verbose:
                print(f"\nDelay {delay}s, trial {trial+1}/{n_trials}")
            
            result = run_single_trial(delay, isotope, verbose=False)
            result['trial'] = trial
            results.append(result)
            
            if verbose:
                print(f"  Eligibility at plateau: {result['eligibility_before_plateau']:.3f}")
                print(f"  Strength change: {result['normalized_change']*100:.1f}%")
    
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze delay sweep results
    """
    import pandas as pd
    
    # Convert to dataframe for easy grouping
    df = pd.DataFrame(results)
    
    # Group by delay and average
    grouped = df.groupby('delay_s').agg({
        'eligibility_before_plateau': ['mean', 'std'],
        'normalized_change': ['mean', 'std'],
    }).reset_index()
    
    delays = grouped['delay_s'].values
    eligibilities = grouped['eligibility_before_plateau']['mean'].values
    eligibility_std = grouped['eligibility_before_plateau']['std'].values
    strength_changes = grouped['normalized_change']['mean'].values
    
    # Find half-decay point
    if eligibilities[0] > 0:
        half_target = 0.5 * eligibilities[0]
        below_half = np.where(eligibilities < half_target)[0]
        if len(below_half) > 0 and below_half[0] > 0:
            idx = below_half[0]
            d1, d2 = delays[idx-1], delays[idx]
            e1, e2 = eligibilities[idx-1], eligibilities[idx]
            half_decay_delay = d1 + (half_target - e1) * (d2 - d1) / (e2 - e1)
        else:
            half_decay_delay = np.nan
    else:
        half_decay_delay = np.nan
    
    return {
        'delays': delays,
        'eligibilities': eligibilities,
        'eligibility_std': eligibility_std,
        'strength_changes': strength_changes,
        'half_decay_delay': half_decay_delay,
        'T2': results[0]['T2'],
        'isotope': results[0]['isotope'],
    }


def plot_comparison(p31_analysis: Dict, p32_analysis: Dict, save_path: str = None):
    """
    Create publication figure comparing P31 vs P32
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === Panel A: Eligibility decay ===
    ax1 = axes[0]
    
    # P31 data
    ax1.semilogy(p31_analysis['delays'], p31_analysis['eligibilities'],
                'o-', color='#2E86AB', markersize=8, linewidth=2, label='³¹P (T₂=67.6s)')
    
    # P31 theory
    t = np.linspace(0, 120, 100)
    e0 = p31_analysis['eligibilities'][0]
    ax1.semilogy(t, e0 * np.exp(-t/67.6), '--', color='#2E86AB', alpha=0.5)
    
    # P32 inset
    ax1_inset = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
    ax1_inset.semilogy(p32_analysis['delays'], p32_analysis['eligibilities'],
                      's-', color='#E94F37', markersize=6, linewidth=2)
    t_p32 = np.linspace(0, 2, 50)
    e0_p32 = p32_analysis['eligibilities'][0]
    ax1_inset.semilogy(t_p32, e0_p32 * np.exp(-t_p32/0.3), '--', color='#E94F37', alpha=0.5)
    ax1_inset.set_xlabel('Delay (s)', fontsize=9)
    ax1_inset.set_ylabel('Eligibility', fontsize=9)
    ax1_inset.set_title('³²P (T₂=0.3s)', fontsize=9)
    ax1_inset.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax1.set_ylabel('Eligibility Level', fontsize=12)
    ax1.set_title('A. Eligibility Trace Decay', fontsize=14)
    ax1.axhline(0.3, color='gray', linestyle=':', alpha=0.5, label='Threshold')
    ax1.legend(loc='lower left')
    ax1.set_xlim(0, 125)
    ax1.set_ylim(0.01, 1)
    ax1.grid(True, alpha=0.3)
    
    # === Panel B: Synaptic strength change ===
    ax2 = axes[1]
    
    ax2.bar(p31_analysis['delays'] - 2, p31_analysis['strength_changes'] * 100,
           width=4, color='#2E86AB', alpha=0.8, label='³¹P')
    
    ax2.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax2.set_ylabel('Synaptic Strength Change (%)', fontsize=12)
    ax2.set_title('B. Plasticity vs Delay', fontsize=14)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Summary text
    summary = (
        f"³¹P half-decay: {p31_analysis['half_decay_delay']:.1f}s "
        f"(expected: {67.6 * np.log(2):.1f}s)\n"
        f"³²P half-decay: {p32_analysis['half_decay_delay']:.2f}s "
        f"(expected: {0.3 * np.log(2):.2f}s)"
    )
    fig.text(0.5, 0.02, summary, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig


def main():
    """Run the full experiment"""
    
    print("="*60)
    print("INTERPULSE DELAY EXPERIMENT - MODEL 6")
    print("="*60)
    
    # P31 delays (long range - eligibility persists)
    delays_p31 = [5, 30, 60]
    n_trials = 1 # Average over trials
    
    # P32 delays (short range - eligibility collapses fast)
    delays_p32 = [0.1, 0.3, 0.5]
    
    # Run P31
    results_p31 = run_delay_sweep('P31', delays_p31, n_trials=1)
    analysis_p31 = analyze_results(results_p31)
    
    # Run P32
    results_p32 = run_delay_sweep('P32', delays_p32, n_trials=1)
    analysis_p32 = analyze_results(results_p32)
    
    # Print key results
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)
    print(f"\n³¹P (quantum coherent):")
    print(f"  Half-decay delay: {analysis_p31['half_decay_delay']:.1f} s")
    print(f"  Expected (T₂ × ln2): {67.6 * np.log(2):.1f} s")
    
    print(f"\n³²P (control):")
    print(f"  Half-decay delay: {analysis_p32['half_decay_delay']:.2f} s")
    print(f"  Expected (T₂ × ln2): {0.3 * np.log(2):.2f} s")
    
    ratio = analysis_p31['half_decay_delay'] / analysis_p32['half_decay_delay']
    print(f"\nRatio (P31/P32): {ratio:.0f}x")
    print(f"Expected ratio: {67.6/0.3:.0f}x")
    
    # Save results
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save figure
    fig_path = output_dir / f"interpulse_delay_experiment_{timestamp}.png"
    fig = plot_comparison(analysis_p31, analysis_p32, save_path=str(fig_path))
    
    # Save data
    data_path = output_dir / f"interpulse_delay_experiment_{timestamp}.json"

    results_dict = {
        'p31_delays': analysis_p31['delays'].tolist(),
        'p31_eligibilities': analysis_p31['eligibilities'].tolist(),
        'p31_strength_changes': analysis_p31['strength_changes'].tolist(),
        'p31_half_decay': float(analysis_p31['half_decay_delay']) if not np.isnan(analysis_p31['half_decay_delay']) else None,
        'p31_T2': float(analysis_p31['T2']),
        'p32_delays': analysis_p32['delays'].tolist(),
        'p32_eligibilities': analysis_p32['eligibilities'].tolist(),
        'p32_strength_changes': analysis_p32['strength_changes'].tolist(),
        'p32_half_decay': float(analysis_p32['half_decay_delay']) if not np.isnan(analysis_p32['half_decay_delay']) else None,
        'p32_T2': float(analysis_p32['T2']),
    }

    with open(data_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    plt.show()
    
    print("\n✓ Experiment complete")
    
    return analysis_p31, analysis_p32


if __name__ == "__main__":
    main()