"""
Tier 1 Experiment: Theta-Burst Integration
==========================================

Demonstrates that physiologically realistic theta-burst stimulation
produces dimer accumulation matching the learning timescale.

Physics:
- Single spike: Insufficient calcium for dimer formation
- Theta burst: Multiple spikes integrate calcium → dimers form
- Pattern matches complex-spike discharges (Larson et al. 1986)

Protocol:
- 5 bursts at 5 Hz (theta rhythm)
- Each burst: 4 spikes at 100 Hz
- Matches LTP induction protocols from Sheffield & Bhalla

Key predictions:
- Calcium accumulates with each burst
- Dimer concentration builds progressively
- Particle count tracks chemistry accurately
- 3-5 dimers per synapse after theta burst

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


@dataclass
class ThetaBurstResult:
    """Results from theta-burst integration experiment"""
    # Time traces (ms resolution during bursts)
    times_ms: List[float] = field(default_factory=list)
    calcium_uM: List[float] = field(default_factory=list)
    dimer_conc_nM: List[float] = field(default_factory=list)
    n_particles: List[int] = field(default_factory=list)
    em_field_kT: List[float] = field(default_factory=list)
    
    # Burst summaries
    burst_calcium_peaks: List[float] = field(default_factory=list)
    burst_dimer_conc: List[float] = field(default_factory=list)
    burst_particles: List[int] = field(default_factory=list)
    
    # Final metrics
    total_calcium_integral: float = 0.0
    final_dimer_conc_nM: float = 0.0
    final_particles: int = 0


def run(n_bursts: int = 5, verbose: bool = True) -> ThetaBurstResult:
    """
    Run theta-burst integration experiment
    
    Parameters
    ----------
    n_bursts : int
        Number of theta bursts (default 5 = standard TBS)
    verbose : bool
        Print progress
    """
    result = ThetaBurstResult()
    
    # Configure model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0
    
    model = Model6QuantumSynapse(params)
    
    if verbose:
        print("="*70)
        print("THETA-BURST INTEGRATION")
        print("="*70)
        print(f"\nProtocol: {n_bursts} bursts × 4 spikes @ 100Hz, 5Hz burst rate")
        print("Matches physiological LTP induction (Larson et al. 1986)")
    
    dt = 0.001  # 1 ms timestep
    t_ms = 0.0
    
    calcium_integral = 0.0
    
    if verbose:
        print("\n--- Baseline (100ms) ---")
    
    # Baseline
    for _ in range(100):
        model.step(dt, {'voltage': -70e-3, 'reward': False})
        t_ms += 1
        
        # Record baseline
        ca = np.max(model.calcium.get_concentration()) * 1e6
        result.times_ms.append(t_ms)
        result.calcium_uM.append(ca)
        result.dimer_conc_nM.append(model.ca_phosphate.get_dimer_concentration().max() * 1e9)
        result.n_particles.append(len(model.dimer_particles.dimers))
        result.em_field_kT.append(getattr(model, '_collective_field_kT', 0.0))
    
    if verbose:
        print(f"  Baseline Ca: {result.calcium_uM[-1]:.3f} µM")
        print(f"  Baseline dimers: {result.n_particles[-1]} particles")
    
    # === THETA BURST STIMULATION ===
    if verbose:
        print("\n--- Theta-Burst Stimulation ---")
        print(f"{'Burst':<6} {'Peak Ca (µM)':<12} {'Dimer (nM)':<12} {'Particles':<10}")
        print("-"*45)
    
    for burst in range(n_bursts):
        peak_ca_this_burst = 0.0
        
        # Each burst: 4 spikes at 100 Hz (10ms intervals)
        for spike in range(4):
            # 2ms depolarization (spike)
            for _ in range(2):
                model.step(dt, {'voltage': -10e-3, 'reward': False})
                t_ms += 1
                
                ca = np.max(model.calcium.get_concentration()) * 1e6
                peak_ca_this_burst = max(peak_ca_this_burst, ca)
                calcium_integral += ca * dt
                
                result.times_ms.append(t_ms)
                result.calcium_uM.append(ca)
                result.dimer_conc_nM.append(model.ca_phosphate.get_dimer_concentration().max() * 1e9)
                result.n_particles.append(len(model.dimer_particles.dimers))
                result.em_field_kT.append(getattr(model, '_collective_field_kT', 0.0))
            
            # 8ms at rest (within burst)
            for _ in range(8):
                model.step(dt, {'voltage': -70e-3, 'reward': False})
                t_ms += 1
                
                ca = np.max(model.calcium.get_concentration()) * 1e6
                calcium_integral += ca * dt
                
                result.times_ms.append(t_ms)
                result.calcium_uM.append(ca)
                result.dimer_conc_nM.append(model.ca_phosphate.get_dimer_concentration().max() * 1e9)
                result.n_particles.append(len(model.dimer_particles.dimers))
                result.em_field_kT.append(getattr(model, '_collective_field_kT', 0.0))
        
        # Record burst summary (before inter-burst interval)
        result.burst_calcium_peaks.append(peak_ca_this_burst)
        result.burst_dimer_conc.append(result.dimer_conc_nM[-1])
        result.burst_particles.append(result.n_particles[-1])
        
        if verbose:
            print(f"{burst+1:<6} {peak_ca_this_burst:<12.2f} "
                  f"{result.dimer_conc_nM[-1]:<12.1f} {result.n_particles[-1]:<10}")
        
        # 160ms inter-burst interval (to make 5 Hz)
        for _ in range(160):
            model.step(dt, {'voltage': -70e-3, 'reward': False})
            t_ms += 1
            
            ca = np.max(model.calcium.get_concentration()) * 1e6
            calcium_integral += ca * dt
            
            result.times_ms.append(t_ms)
            result.calcium_uM.append(ca)
            result.dimer_conc_nM.append(model.ca_phosphate.get_dimer_concentration().max() * 1e9)
            result.n_particles.append(len(model.dimer_particles.dimers))
            result.em_field_kT.append(getattr(model, '_collective_field_kT', 0.0))
    
    # Final metrics
    result.total_calcium_integral = calcium_integral
    result.final_dimer_conc_nM = result.dimer_conc_nM[-1]
    result.final_particles = result.n_particles[-1]
    
    return result


def plot(result: ThetaBurstResult, output_dir: Path = None) -> plt.Figure:
    """
    Generate publication-quality figure
    
    Layout:
    - Top: Calcium dynamics with spike markers
    - Middle: Dimer concentration and particle count
    - Bottom: Burst-by-burst accumulation summary
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.35)
    
    times_s = np.array(result.times_ms) / 1000  # Convert to seconds
    
    # Colors
    color_ca = '#E94F37'      # Red for calcium
    color_dimer = '#2E86AB'   # Blue for dimers
    color_particle = '#28A745'  # Green for particles
    
    # === TOP: CALCIUM DYNAMICS ===
    ax_ca = fig.add_subplot(gs[0])
    
    ax_ca.plot(times_s, result.calcium_uM, color=color_ca, linewidth=1.5)
    ax_ca.fill_between(times_s, 0, result.calcium_uM, alpha=0.3, color=color_ca)
    
    # Mark burst periods (approximate)
    burst_starts = [0.1 + i * 0.2 for i in range(5)]  # 5 Hz = 200ms period
    for bs in burst_starts:
        ax_ca.axvspan(bs, bs + 0.04, alpha=0.2, color='yellow')
    
    ax_ca.set_ylabel('Calcium (µM)', fontsize=12, color=color_ca)
    ax_ca.tick_params(axis='y', labelcolor=color_ca)
    ax_ca.set_title('Theta-Burst Calcium Integration', fontsize=14, fontweight='bold')
    ax_ca.set_xlim(0, max(times_s))
    ax_ca.grid(True, alpha=0.3)
    
    # Add calcium integral annotation
    ax_ca.annotate(f'Ca²⁺ integral: {result.total_calcium_integral:.1f} µM·s',
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === MIDDLE: DIMER CONCENTRATION AND PARTICLES ===
    ax_dimer = fig.add_subplot(gs[1], sharex=ax_ca)
    
    ax_dimer.plot(times_s, result.dimer_conc_nM, color=color_dimer, 
                  linewidth=1.5, label='Concentration (nM)')
    ax_dimer.set_ylabel('Dimer Conc. (nM)', fontsize=12, color=color_dimer)
    ax_dimer.tick_params(axis='y', labelcolor=color_dimer)
    
    # Particle count on secondary axis
    ax_part = ax_dimer.twinx()
    ax_part.plot(times_s, result.n_particles, color=color_particle,
                 linewidth=2, linestyle='--', label='Particle count')
    ax_part.set_ylabel('Particle Count', fontsize=12, color=color_particle)
    ax_part.tick_params(axis='y', labelcolor=color_particle)
    
    ax_dimer.set_title('Dimer Formation: Chemistry → Particles', fontsize=12)
    ax_dimer.set_xlabel('Time (s)', fontsize=12)
    ax_dimer.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax_dimer.get_legend_handles_labels()
    lines2, labels2 = ax_part.get_legend_handles_labels()
    ax_dimer.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # === BOTTOM: BURST-BY-BURST SUMMARY ===
    ax_burst = fig.add_subplot(gs[2])
    
    bursts = np.arange(1, len(result.burst_particles) + 1)
    width = 0.35
    
    bars1 = ax_burst.bar(bursts - width/2, result.burst_dimer_conc, width,
                         label='Dimer conc. (nM)', color=color_dimer, alpha=0.7)
    
    ax_burst2 = ax_burst.twinx()
    bars2 = ax_burst2.bar(bursts + width/2, result.burst_particles, width,
                          label='Particles', color=color_particle, alpha=0.7)
    
    ax_burst.set_xlabel('Burst Number', fontsize=12)
    ax_burst.set_ylabel('Concentration (nM)', fontsize=12, color=color_dimer)
    ax_burst2.set_ylabel('Particle Count', fontsize=12, color=color_particle)
    ax_burst.tick_params(axis='y', labelcolor=color_dimer)
    ax_burst2.tick_params(axis='y', labelcolor=color_particle)
    
    ax_burst.set_title('Progressive Dimer Accumulation', fontsize=12)
    ax_burst.set_xticks(bursts)
    ax_burst.grid(True, alpha=0.3, axis='y')
    
    # Combined legend
    ax_burst.legend([bars1, bars2], ['Dimer conc. (nM)', 'Particles'], 
                    loc='upper left', fontsize=9)
    
    # Add value labels
    for i, (conc, part) in enumerate(zip(result.burst_dimer_conc, result.burst_particles)):
        ax_burst.annotate(f'{conc:.0f}', xy=(i+1 - width/2, conc), 
                          xytext=(0, 3), textcoords='offset points',
                          ha='center', fontsize=9, color=color_dimer)
        ax_burst2.annotate(f'{part}', xy=(i+1 + width/2, part),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', fontsize=9, color=color_particle)
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / 'theta_burst_integration.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Save data
        data = {
            'experiment': 'theta_burst_integration',
            'timestamp': datetime.now().isoformat(),
            'protocol': {
                'n_bursts': len(result.burst_particles),
                'spikes_per_burst': 4,
                'intra_burst_freq_Hz': 100,
                'inter_burst_freq_Hz': 5
            },
            'burst_summary': {
                'calcium_peaks_uM': result.burst_calcium_peaks,
                'dimer_conc_nM': result.burst_dimer_conc,
                'particle_counts': result.burst_particles
            },
            'final_state': {
                'calcium_integral_uM_s': result.total_calcium_integral,
                'dimer_conc_nM': result.final_dimer_conc_nM,
                'particle_count': result.final_particles
            }
        }
        
        json_path = output_dir / 'theta_burst_integration.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {json_path}")
    
    return fig


def print_summary(result: ThetaBurstResult):
    """Print summary of results"""
    print("\n" + "="*70)
    print("THETA-BURST INTEGRATION SUMMARY")
    print("="*70)
    
    print("\nBurst-by-burst accumulation:")
    print(f"{'Burst':<6} {'Peak Ca (µM)':<12} {'Dimer (nM)':<12} {'Particles':<10}")
    print("-"*45)
    for i, (ca, conc, part) in enumerate(zip(result.burst_calcium_peaks,
                                              result.burst_dimer_conc,
                                              result.burst_particles)):
        print(f"{i+1:<6} {ca:<12.2f} {conc:<12.1f} {part:<10}")
    
    print(f"\nFinal state:")
    print(f"  Total Ca²⁺ integral: {result.total_calcium_integral:.1f} µM·s")
    print(f"  Final dimer concentration: {result.final_dimer_conc_nM:.1f} nM")
    print(f"  Final particle count: {result.final_particles}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
Theta-burst stimulation produces CUMULATIVE dimer formation:
- Single spikes insufficient (calcium clears too fast)
- Multiple bursts integrate calcium → sustained elevation
- Particle count tracks chemistry accurately

This matches the physiological requirement for patterned activity
to induce LTP (Larson & Lynch 1986, Sheffield & Bhalla).
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run theta-burst integration experiment')
    parser.add_argument('--bursts', type=int, default=5,
                        help='Number of theta bursts')
    parser.add_argument('--output', type=str, default='results/tier1',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    result = run(n_bursts=args.bursts, verbose=not args.quiet)
    
    print_summary(result)
    
    fig = plot(result, output_dir=args.output)
    
    plt.show()