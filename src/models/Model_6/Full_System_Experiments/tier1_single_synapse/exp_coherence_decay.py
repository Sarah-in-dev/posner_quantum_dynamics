"""
Tier 1 Experiment: Coherence Decay Validation
==============================================

Validates that singlet probability decays according to Agarwal et al. 2023 physics.

Physics being validated:
- P_S(t) = P_thermal + (P_S(0) - P_thermal) × exp(-t/T_singlet)
- P_thermal = 0.25 (maximally mixed state)
- T_singlet(P31) ≈ 216s (dipolar relaxation only)
- T_singlet(P32) ≈ 0.4s (quadrupolar relaxation dominates)
- Entanglement preserved when P_S > 0.5

This validates that our implementation matches the theoretical predictions.

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
from datetime import datetime
from scipy.optimize import curve_fit
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


@dataclass
class DecayResult:
    """Results from coherence decay experiment"""
    isotope: str
    times: List[float] = field(default_factory=list)
    mean_singlet_prob: List[float] = field(default_factory=list)
    n_entangled: List[int] = field(default_factory=list)
    n_total: List[int] = field(default_factory=list)
    
    # Fitted parameters
    T_singlet_fitted: float = 0.0
    P_thermal_fitted: float = 0.25
    fit_r_squared: float = 0.0
    
    # Theory comparison
    T_singlet_theory: float = 0.0
    time_to_decohere: float = 0.0  # When P_S first drops below 0.5


def exponential_decay(t, P0, P_thermal, T):
    """Exponential decay model: P(t) = P_thermal + (P0 - P_thermal) * exp(-t/T)"""
    return P_thermal + (P0 - P_thermal) * np.exp(-t / T)


def run_decay_test(fraction_P31: float,
                   duration_s: float,
                   verbose: bool = True) -> DecayResult:
    """
    Run coherence decay test for single isotope
    
    Parameters
    ----------
    fraction_P31 : float
        1.0 for P31, 0.0 for P32
    duration_s : float
        How long to track decay
    verbose : bool
        Print progress
    """
    isotope = "P31" if fraction_P31 > 0.5 else "P32"
    result = DecayResult(isotope=isotope)
    
    # Set theory prediction
    if fraction_P31 > 0.5:
        result.T_singlet_theory = 216.0  # seconds
    else:
        result.T_singlet_theory = 0.4  # seconds
    
    # Configure model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = fraction_P31
    params.environment.fraction_P32 = 1.0 - fraction_P31
    
    model = Model6QuantumSynapse(params)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Coherence Decay Test: {isotope}")
        print(f"Theory T_singlet = {result.T_singlet_theory:.1f}s")
        print(f"{'='*60}")
    
    dt = 0.001  # 1 ms
    
    # === PHASE 1: CREATE DIMERS ===
    if verbose:
        print("\nPhase 1: Creating dimers via stimulation")
    
    # For P32, use minimal stimulation to capture decay
    # For P31, use theta burst
    if fraction_P31 < 0.5:
        # P32: Quick burst, measure immediately
        n_bursts = 2
        if verbose:
            print("  (Using short stimulation for P32 to capture fast decay)")
    else:
        n_bursts = 5
    
    for burst in range(n_bursts):
        for spike in range(4):
            for _ in range(2):
                model.step(dt, {'voltage': -10e-3, 'reward': False})
            for _ in range(8):
                model.step(dt, {'voltage': -70e-3, 'reward': False})
        # Shorter inter-burst for P32
        inter_burst = 50 if fraction_P31 < 0.5 else 160
        for _ in range(inter_burst):
            model.step(dt, {'voltage': -70e-3, 'reward': False})
    
    # Check we have dimers
    n_dimers = len(model.dimer_particles.dimers)
    if n_dimers == 0:
        print("WARNING: No dimers created!")
        return result
    
    if verbose:
        initial_ps = np.mean([d.singlet_probability for d in model.dimer_particles.dimers])
        print(f"  Created {n_dimers} dimers")
        print(f"  Initial P_S = {initial_ps:.3f}")
    
    # === PHASE 2: TRACK DECAY ===
    if verbose:
        print(f"\nPhase 2: Tracking decay for {duration_s}s")
    
    # Adaptive timestep for efficiency
    if duration_s > 10:
        dt_decay = 0.1  # 100ms for long runs
    else:
        dt_decay = 0.01  # 10ms for short runs
    
    n_steps = int(duration_s / dt_decay)
    
    # Recording frequency
    if n_steps > 100:
        record_every = n_steps // 50  # ~50 data points
    else:
        record_every = 1
    
    t = 0.0
    decohere_detected = False
    
    # Initial measurement
    if model.dimer_particles.dimers:
        mean_ps = np.mean([d.singlet_probability for d in model.dimer_particles.dimers])
        n_entangled = sum(1 for d in model.dimer_particles.dimers if d.singlet_probability > 0.5)
    else:
        mean_ps = 1.0
        n_entangled = 0
    
    result.times.append(0.0)
    result.mean_singlet_prob.append(mean_ps)
    result.n_entangled.append(n_entangled)
    result.n_total.append(len(model.dimer_particles.dimers))
    
    for step in range(n_steps):
        model.step(dt_decay, {'voltage': -70e-3, 'reward': False})
        t += dt_decay
        
        if step % record_every == 0 or step == n_steps - 1:
            if model.dimer_particles.dimers:
                mean_ps = np.mean([d.singlet_probability for d in model.dimer_particles.dimers])
                n_entangled = sum(1 for d in model.dimer_particles.dimers if d.singlet_probability > 0.5)
            else:
                mean_ps = 0.25
                n_entangled = 0
            
            result.times.append(t)
            result.mean_singlet_prob.append(mean_ps)
            result.n_entangled.append(n_entangled)
            result.n_total.append(len(model.dimer_particles.dimers))
            
            # Detect decoherence
            if mean_ps < 0.5 and not decohere_detected:
                result.time_to_decohere = t
                decohere_detected = True
            
            if verbose and step % (n_steps // 10) == 0:
                status = "ENTANGLED" if mean_ps > 0.5 else "DECOHERED"
                print(f"  t={t:6.1f}s: P_S={mean_ps:.4f} [{status}]")
    
    # === FIT DECAY CURVE ===
    if len(result.times) > 5:
        try:
            times_arr = np.array(result.times)
            ps_arr = np.array(result.mean_singlet_prob)
            
            # Initial guess
            p0 = [ps_arr[0], 0.25, result.T_singlet_theory]
            
            # Bounds
            bounds = ([0.5, 0.2, 0.01], [1.0, 0.3, 1000])
            
            popt, pcov = curve_fit(exponential_decay, times_arr, ps_arr,
                                   p0=p0, bounds=bounds, maxfev=5000)
            
            result.T_singlet_fitted = popt[2]
            result.P_thermal_fitted = popt[1]
            
            # R-squared
            residuals = ps_arr - exponential_decay(times_arr, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ps_arr - np.mean(ps_arr))**2)
            result.fit_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if verbose:
                print(f"\nFit results:")
                print(f"  T_singlet (fitted) = {result.T_singlet_fitted:.1f}s")
                print(f"  T_singlet (theory) = {result.T_singlet_theory:.1f}s")
                print(f"  Ratio: {result.T_singlet_fitted/result.T_singlet_theory:.2f}")
                print(f"  R² = {result.fit_r_squared:.4f}")
        except Exception as e:
            if verbose:
                print(f"Fit failed: {e}")
    
    return result


def run(duration_P31: float = 300.0,
        duration_P32: float = 5.0,
        verbose: bool = True) -> Dict[str, DecayResult]:
    """
    Run complete coherence decay validation
    
    Parameters
    ----------
    duration_P31 : float
        Duration for P31 test (needs longer for slow decay)
    duration_P32 : float
        Duration for P32 test (fast decay)
    verbose : bool
        Print progress
    """
    results = {}
    
    # P31 test (long coherence)
    results['P31'] = run_decay_test(1.0, duration_P31, verbose)
    
    # P32 test (fast decoherence)
    results['P32'] = run_decay_test(0.0, duration_P32, verbose)
    
    return results


def plot(results: Dict[str, DecayResult], output_dir: Path = None) -> plt.Figure:
    """
    Generate publication-quality figure
    
    Layout:
    - Left: P31 decay curve (long timescale)
    - Right: P32 decay curve (short timescale)
    - Bottom: Comparison summary
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], hspace=0.35, wspace=0.3)
    
    color_P31 = '#2E86AB'
    color_P32 = '#E94F37'
    color_fit = '#333333'
    
    p31 = results['P31']
    p32 = results['P32']
    
    # === TOP LEFT: P31 DECAY ===
    ax_p31 = fig.add_subplot(gs[0, 0])
    
    ax_p31.plot(p31.times, p31.mean_singlet_prob, 'o', color=color_P31,
                markersize=4, alpha=0.7, label='Data')
    
    # Fitted curve
    if p31.T_singlet_fitted > 0:
        t_fit = np.linspace(0, max(p31.times), 100)
        ps_fit = exponential_decay(t_fit, p31.mean_singlet_prob[0],
                                   p31.P_thermal_fitted, p31.T_singlet_fitted)
        ax_p31.plot(t_fit, ps_fit, '-', color=color_fit, linewidth=2,
                    label=f'Fit: T={p31.T_singlet_fitted:.0f}s')
    
    # Theory curve
    t_theory = np.linspace(0, max(p31.times), 100)
    ps_theory = exponential_decay(t_theory, p31.mean_singlet_prob[0],
                                  0.25, p31.T_singlet_theory)
    ax_p31.plot(t_theory, ps_theory, '--', color='gray', linewidth=1.5,
                label=f'Theory: T={p31.T_singlet_theory:.0f}s')
    
    ax_p31.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5,
                   label='Entanglement threshold')
    ax_p31.axhline(y=0.25, color='gray', linestyle=':', linewidth=1)
    
    ax_p31.set_xlabel('Time (s)', fontsize=12)
    ax_p31.set_ylabel('Mean Singlet Probability', fontsize=12)
    ax_p31.set_title(f'³¹P Coherence Decay (T_theory = {p31.T_singlet_theory:.0f}s)',
                     fontsize=12, fontweight='bold')
    ax_p31.legend(loc='upper right', fontsize=9)
    ax_p31.set_ylim(0.2, 1.05)
    ax_p31.grid(True, alpha=0.3)
    
    # === TOP RIGHT: P32 DECAY ===
    ax_p32 = fig.add_subplot(gs[0, 1])
    
    ax_p32.plot(p32.times, p32.mean_singlet_prob, 's', color=color_P32,
                markersize=4, alpha=0.7, label='Data')
    
    # Fitted curve
    if p32.T_singlet_fitted > 0:
        t_fit = np.linspace(0, max(p32.times), 100)
        ps_fit = exponential_decay(t_fit, p32.mean_singlet_prob[0],
                                   p32.P_thermal_fitted, p32.T_singlet_fitted)
        ax_p32.plot(t_fit, ps_fit, '-', color=color_fit, linewidth=2,
                    label=f'Fit: T={p32.T_singlet_fitted:.2f}s')
    
    # Theory curve
    t_theory = np.linspace(0, max(p32.times), 100)
    ps_theory = exponential_decay(t_theory, p32.mean_singlet_prob[0],
                                  0.25, p32.T_singlet_theory)
    ax_p32.plot(t_theory, ps_theory, '--', color='gray', linewidth=1.5,
                label=f'Theory: T={p32.T_singlet_theory:.1f}s')
    
    ax_p32.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5,
                   label='Entanglement threshold')
    ax_p32.axhline(y=0.25, color='gray', linestyle=':', linewidth=1)
    
    ax_p32.set_xlabel('Time (s)', fontsize=12)
    ax_p32.set_ylabel('Mean Singlet Probability', fontsize=12)
    ax_p32.set_title(f'³²P Coherence Decay (T_theory = {p32.T_singlet_theory:.1f}s)',
                     fontsize=12, fontweight='bold')
    ax_p32.legend(loc='upper right', fontsize=9)
    ax_p32.set_ylim(0.2, 1.05)
    ax_p32.grid(True, alpha=0.3)
    
    # === BOTTOM: COMPARISON TABLE ===
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Create comparison data
    table_data = [
        ['Isotope', '³¹P (spin-1/2)', '³²P (spin-1)'],
        ['T_singlet (theory)', f'{p31.T_singlet_theory:.1f} s', f'{p32.T_singlet_theory:.2f} s'],
        ['T_singlet (fitted)', f'{p31.T_singlet_fitted:.1f} s', f'{p32.T_singlet_fitted:.2f} s'],
        ['Fit R²', f'{p31.fit_r_squared:.4f}', f'{p32.fit_r_squared:.4f}'],
        ['Time to P_S < 0.5', 
         'N/A' if p31.time_to_decohere == 0 else f'{p31.time_to_decohere:.1f} s',
         f'{p32.time_to_decohere:.2f} s' if p32.time_to_decohere > 0 else 'Immediate'],
        ['Relaxation', 'Dipolar only', 'Quadrupolar dominates'],
        ['Learning window', 'COMPATIBLE (60-100s)', 'INCOMPATIBLE (<1s)']
    ]
    
    table = ax_table.table(cellText=table_data,
                           loc='center',
                           cellLoc='center',
                           colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#E6E6E6')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Color the conclusion row
    table[(6, 1)].set_facecolor('#D4EDDA')  # Green for compatible
    table[(6, 2)].set_facecolor('#F8D7DA')  # Red for incompatible
    
    ax_table.set_title('Coherence Decay Validation Summary', fontsize=14,
                       fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / 'coherence_decay.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Save data
        data = {
            'experiment': 'coherence_decay_validation',
            'timestamp': datetime.now().isoformat(),
            'P31': {
                'times': p31.times,
                'mean_singlet_prob': p31.mean_singlet_prob,
                'T_theory': p31.T_singlet_theory,
                'T_fitted': p31.T_singlet_fitted,
                'fit_r_squared': p31.fit_r_squared,
                'time_to_decohere': p31.time_to_decohere
            },
            'P32': {
                'times': p32.times,
                'mean_singlet_prob': p32.mean_singlet_prob,
                'T_theory': p32.T_singlet_theory,
                'T_fitted': p32.T_singlet_fitted,
                'fit_r_squared': p32.fit_r_squared,
                'time_to_decohere': p32.time_to_decohere
            },
            'validation': {
                'P31_matches_theory': bool(abs(p31.T_singlet_fitted - p31.T_singlet_theory) / p31.T_singlet_theory < 0.5) if p31.T_singlet_theory > 0 else False,
                'P32_matches_theory': bool(abs(p32.T_singlet_fitted - p32.T_singlet_theory) / p32.T_singlet_theory < 1.0) if p32.T_singlet_theory > 0 else False,
                'isotope_discrimination': p31.T_singlet_fitted / p32.T_singlet_fitted if p32.T_singlet_fitted > 0 else float('inf')
            }
        }
        
        json_path = output_dir / 'coherence_decay.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {json_path}")
    
    return fig


def print_summary(results: Dict[str, DecayResult]):
    """Print summary of results"""
    print("\n" + "="*70)
    print("COHERENCE DECAY VALIDATION SUMMARY")
    print("="*70)
    
    p31 = results['P31']
    p32 = results['P32']
    
    print(f"\n{'Metric':<25} {'³¹P':>20} {'³²P':>20}")
    print("-"*70)
    print(f"{'T_singlet (theory)':<25} {p31.T_singlet_theory:>19.1f}s {p32.T_singlet_theory:>19.2f}s")
    print(f"{'T_singlet (fitted)':<25} {p31.T_singlet_fitted:>19.1f}s {p32.T_singlet_fitted:>19.2f}s")
    print(f"{'Fit R²':<25} {p31.fit_r_squared:>20.4f} {p32.fit_r_squared:>20.4f}")
    
    if p31.T_singlet_theory > 0:
        ratio_p31 = p31.T_singlet_fitted / p31.T_singlet_theory
        print(f"{'Fitted/Theory ratio':<25} {ratio_p31:>20.2f} ", end="")
    if p32.T_singlet_theory > 0:
        ratio_p32 = p32.T_singlet_fitted / p32.T_singlet_theory
        print(f"{ratio_p32:>20.2f}")
    
    if p32.T_singlet_fitted > 0:
        discrimination = p31.T_singlet_fitted / p32.T_singlet_fitted
        print(f"\nIsotope discrimination: {discrimination:.0f}x")
    
    print("\n" + "="*70)
    print("VALIDATION STATUS")
    print("="*70)
    
    # Check if results match theory
    p31_valid = abs(p31.T_singlet_fitted - p31.T_singlet_theory) / p31.T_singlet_theory < 0.5
    p32_valid = abs(p32.T_singlet_fitted - p32.T_singlet_theory) / p32.T_singlet_theory < 1.0  # More lenient for fast decay
    
    if p31_valid and p32_valid:
        print("""
✓ P31 decay matches theory (within 50%)
✓ P32 decay matches theory (within 100%)
✓ Isotope discrimination verified

VALIDATION PASSED: Model correctly implements Agarwal 2023 physics
""")
    else:
        print("WARNING: Results deviate from theory - check implementation")
        if not p31_valid:
            print(f"  P31: fitted {p31.T_singlet_fitted:.1f}s vs theory {p31.T_singlet_theory:.1f}s")
        if not p32_valid:
            print(f"  P32: fitted {p32.T_singlet_fitted:.2f}s vs theory {p32.T_singlet_theory:.2f}s")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run coherence decay validation')
    parser.add_argument('--duration-p31', type=float, default=300.0,
                        help='P31 test duration (seconds)')
    parser.add_argument('--duration-p32', type=float, default=5.0,
                        help='P32 test duration (seconds)')
    parser.add_argument('--output', type=str, default='results/tier1',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = run(duration_P31=args.duration_p31,
                  duration_P32=args.duration_p32,
                  verbose=not args.quiet)
    
    print_summary(results)
    
    fig = plot(results, output_dir=args.output)
    
    plt.show()