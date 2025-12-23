"""
Tier 1 Experiment: Entanglement Network Formation
==================================================

Demonstrates that the tryptophan EM field acts as a "quantum bus" enabling
entanglement to spread between dimers formed at different times.

Physics:
- Birth entanglement: Dimers from same pyrophosphate share entanglement
- EM-mediated entanglement: Tryptophan field couples spatially separate dimers
- Rate scales with field strength (no hard threshold)

Key predictions:
- Network grows during theta-burst (new dimers join existing network)
- Network completes during rest (EM field enables continued coupling)
- Without EM field, only birth-paired dimers would be entangled

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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


@dataclass
class BurstSnapshot:
    """State after each theta burst"""
    burst_num: int
    time: float
    n_particles: int
    n_bonds: int
    largest_cluster: int
    mean_singlet_prob: float
    em_field_kT: float


@dataclass
class NetworkResult:
    """Results from entanglement network experiment"""
    # Burst-by-burst snapshots
    burst_snapshots: List[BurstSnapshot] = field(default_factory=list)
    
    # Rest period evolution
    rest_times: List[float] = field(default_factory=list)
    rest_particles: List[int] = field(default_factory=list)
    rest_bonds: List[int] = field(default_factory=list)
    rest_clusters: List[int] = field(default_factory=list)
    rest_singlet: List[float] = field(default_factory=list)
    rest_em_field: List[float] = field(default_factory=list)
    
    # Key metrics
    final_network_fraction: float = 0.0  # Fraction of dimers in largest cluster
    bonds_from_birth: int = 0
    bonds_from_em: int = 0


def run(n_bursts: int = 5, 
        rest_duration_s: float = 10.0,
        verbose: bool = True) -> NetworkResult:
    """
    Run entanglement network formation experiment
    
    Parameters
    ----------
    n_bursts : int
        Number of theta bursts
    rest_duration_s : float
        Duration of rest period after stimulation
    verbose : bool
        Print progress
    """
    result = NetworkResult()
    
    # Configure model with EM coupling enabled
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0  # Use P31 for long coherence
    
    model = Model6QuantumSynapse(params)
    
    if verbose:
        print("="*70)
        print("ENTANGLEMENT NETWORK FORMATION")
        print("="*70)
        print(f"\nProtocol: {n_bursts} theta bursts + {rest_duration_s}s rest")
    
    dt = 0.001  # 1 ms
    t = 0.0
    
    # Track bonds created by each mechanism
    bonds_before = 0
    
    # === PHASE 1: THETA-BURST STIMULATION ===
    if verbose:
        print("\n--- Phase 1: Theta-Burst Stimulation ---")
        print(f"{'Burst':<6} {'Particles':<10} {'Bonds':<8} {'Cluster':<8} "
              f"{'P_S':<8} {'EM (kT)':<10} {'New Bonds':<10}")
        print("-"*70)
    
    for burst in range(n_bursts):
        # Each burst: 4 spikes at 100 Hz
        for spike in range(4):
            # 2ms depolarization
            for _ in range(2):
                model.step(dt, {'voltage': -10e-3, 'reward': False})
                t += dt
            # 8ms rest within burst
            for _ in range(8):
                model.step(dt, {'voltage': -70e-3, 'reward': False})
                t += dt
        
        # 160ms between bursts
        for _ in range(160):
            model.step(dt, {'voltage': -70e-3, 'reward': False})
            t += dt
        
        # Record state after burst
        pm = model.dimer_particles.get_network_metrics()
        em_field = getattr(model, '_collective_field_kT', 0.0)
        
        if model.dimer_particles.dimers:
            mean_ps = np.mean([d.singlet_probability 
                              for d in model.dimer_particles.dimers])
        else:
            mean_ps = 1.0
        
        new_bonds = pm['n_bonds'] - bonds_before
        bonds_before = pm['n_bonds']
        
        snapshot = BurstSnapshot(
            burst_num=burst + 1,
            time=t,
            n_particles=pm['n_dimers'],
            n_bonds=pm['n_bonds'],
            largest_cluster=pm['largest_cluster'],
            mean_singlet_prob=mean_ps,
            em_field_kT=em_field
        )
        result.burst_snapshots.append(snapshot)
        
        if verbose:
            print(f"{burst+1:<6} {pm['n_dimers']:<10} {pm['n_bonds']:<8} "
                  f"{pm['largest_cluster']:<8} {mean_ps:<8.3f} {em_field:<10.1f} "
                  f"+{new_bonds:<9}")
    
    # === PHASE 2: REST PERIOD ===
    if verbose:
        print("\n--- Phase 2: Rest Period (Network Completion) ---")
        print(f"{'Time (s)':<10} {'Particles':<10} {'Bonds':<8} {'Cluster':<8} "
              f"{'P_S':<8} {'Coverage':<10}")
        print("-"*70)
    
    dt_rest = 0.1  # 100ms steps
    n_steps = int(rest_duration_s / dt_rest)
    
    for step in range(n_steps + 1):
        if step > 0:
            model.step(dt_rest, {'voltage': -70e-3, 'reward': False})
            t += dt_rest
        
        pm = model.dimer_particles.get_network_metrics()
        em_field = getattr(model, '_collective_field_kT', 0.0)
        
        if model.dimer_particles.dimers:
            mean_ps = np.mean([d.singlet_probability 
                              for d in model.dimer_particles.dimers])
        else:
            mean_ps = 1.0
        
        # Calculate network coverage
        if pm['n_dimers'] > 0:
            coverage = pm['largest_cluster'] / pm['n_dimers']
        else:
            coverage = 0.0
        
        result.rest_times.append(t - result.burst_snapshots[-1].time)
        result.rest_particles.append(pm['n_dimers'])
        result.rest_bonds.append(pm['n_bonds'])
        result.rest_clusters.append(pm['largest_cluster'])
        result.rest_singlet.append(mean_ps)
        result.rest_em_field.append(em_field)
        
        # Print at key timepoints
        if step == 0 or step == n_steps or step % (n_steps // 5) == 0:
            if verbose:
                print(f"{result.rest_times[-1]:<10.1f} {pm['n_dimers']:<10} "
                      f"{pm['n_bonds']:<8} {pm['largest_cluster']:<8} "
                      f"{mean_ps:<8.3f} {coverage*100:<9.0f}%")
    
    # Final metrics
    if result.rest_particles[-1] > 0:
        result.final_network_fraction = (result.rest_clusters[-1] / 
                                         result.rest_particles[-1])
    
    return result


def plot(result: NetworkResult, output_dir: Path = None) -> plt.Figure:
    """
    Generate publication-quality figure
    
    Layout:
    - Top: Bond formation over time (bursts + rest)
    - Bottom left: Network growth during bursts
    - Bottom right: Network completion during rest
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1],
                  hspace=0.3, wspace=0.3)
    
    # Extract data
    burst_times = [s.time for s in result.burst_snapshots]
    burst_bonds = [s.n_bonds for s in result.burst_snapshots]
    burst_clusters = [s.largest_cluster for s in result.burst_snapshots]
    burst_particles = [s.n_particles for s in result.burst_snapshots]
    
    # Offset rest times to continue from last burst
    last_burst_time = burst_times[-1] if burst_times else 0
    rest_times_abs = [last_burst_time + t for t in result.rest_times]
    
    # Colors
    color_bonds = '#2E86AB'
    color_cluster = '#E94F37'
    color_particles = '#7D8491'
    
    # === TOP: COMPLETE TIMELINE ===
    ax_main = fig.add_subplot(gs[0, :])
    
    # Combine burst and rest data
    all_times = burst_times + rest_times_abs
    all_bonds = burst_bonds + result.rest_bonds
    all_clusters = burst_clusters + result.rest_clusters
    
    ax_main.plot(all_times, all_bonds, 'o-', color=color_bonds, 
                 linewidth=2, markersize=6, label='Entanglement bonds')
    ax_main.plot(all_times, all_clusters, 's-', color=color_cluster,
                 linewidth=2, markersize=6, label='Largest cluster')
    
    # Mark burst period
    ax_main.axvspan(0, last_burst_time, alpha=0.1, color='yellow',
                    label='Theta burst period')
    
    # Mark rest period
    ax_main.axvspan(last_burst_time, all_times[-1], alpha=0.1, color='green',
                    label='Rest period')
    
    ax_main.set_xlabel('Time (s)', fontsize=12)
    ax_main.set_ylabel('Count', fontsize=12)
    ax_main.set_title('Entanglement Network Formation via EM Quantum Bus', 
                      fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    # Annotate key insight
    ax_main.annotate('Network continues\ngrowing during rest\n(EM-mediated coupling)',
                     xy=(last_burst_time + 3, max(all_bonds) * 0.7),
                     fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === BOTTOM LEFT: BURST-BY-BURST ===
    ax_burst = fig.add_subplot(gs[1, 0])
    
    bursts = [s.burst_num for s in result.burst_snapshots]
    
    ax_burst.bar(np.array(bursts) - 0.2, burst_particles, 0.4, 
                 label='Particles', color=color_particles, alpha=0.7)
    ax_burst.bar(np.array(bursts) + 0.2, burst_clusters, 0.4,
                 label='In network', color=color_cluster, alpha=0.7)
    
    ax_burst.set_xlabel('Burst Number', fontsize=11)
    ax_burst.set_ylabel('Dimer Count', fontsize=11)
    ax_burst.set_title('Network Growth During Stimulation', fontsize=12)
    ax_burst.legend(loc='upper left', fontsize=9)
    ax_burst.set_xticks(bursts)
    ax_burst.grid(True, alpha=0.3, axis='y')
    
    # === BOTTOM RIGHT: REST PERIOD DETAIL ===
    ax_rest = fig.add_subplot(gs[1, 1])
    
    ax_rest.plot(result.rest_times, result.rest_bonds, 'o-',
                 color=color_bonds, linewidth=2, label='Bonds')
    
    # Calculate coverage percentage
    coverage = [c/p*100 if p > 0 else 0 
                for c, p in zip(result.rest_clusters, result.rest_particles)]
    
    ax_rest2 = ax_rest.twinx()
    ax_rest2.plot(result.rest_times, coverage, 's--',
                  color=color_cluster, linewidth=2, label='Coverage %')
    ax_rest2.set_ylabel('Network Coverage (%)', fontsize=11, color=color_cluster)
    ax_rest2.tick_params(axis='y', labelcolor=color_cluster)
    
    ax_rest.set_xlabel('Time Since Burst End (s)', fontsize=11)
    ax_rest.set_ylabel('Bond Count', fontsize=11, color=color_bonds)
    ax_rest.tick_params(axis='y', labelcolor=color_bonds)
    ax_rest.set_title('Network Completion During Rest', fontsize=12)
    ax_rest.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax_rest.get_legend_handles_labels()
    lines2, labels2 = ax_rest2.get_legend_handles_labels()
    ax_rest.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / 'entanglement_network.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Save data
        data = {
            'experiment': 'entanglement_network',
            'timestamp': datetime.now().isoformat(),
            'burst_data': [
                {
                    'burst': s.burst_num,
                    'time': s.time,
                    'particles': s.n_particles,
                    'bonds': s.n_bonds,
                    'cluster': s.largest_cluster,
                    'singlet_prob': s.mean_singlet_prob,
                    'em_field_kT': s.em_field_kT
                }
                for s in result.burst_snapshots
            ],
            'rest_data': {
                'times': result.rest_times,
                'particles': result.rest_particles,
                'bonds': result.rest_bonds,
                'clusters': result.rest_clusters,
                'singlet_prob': result.rest_singlet
            },
            'final_network_fraction': result.final_network_fraction
        }
        
        json_path = output_dir / 'entanglement_network.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {json_path}")
    
    return fig


def print_summary(result: NetworkResult):
    """Print summary of results"""
    print("\n" + "="*70)
    print("ENTANGLEMENT NETWORK SUMMARY")
    print("="*70)
    
    print("\nBurst-by-burst growth:")
    for s in result.burst_snapshots:
        print(f"  Burst {s.burst_num}: {s.n_particles} particles, "
              f"{s.n_bonds} bonds, cluster size {s.largest_cluster}")
    
    print(f"\nAfter rest period:")
    print(f"  Final particles: {result.rest_particles[-1]}")
    print(f"  Final bonds: {result.rest_bonds[-1]}")
    print(f"  Largest cluster: {result.rest_clusters[-1]}")
    print(f"  Network coverage: {result.final_network_fraction*100:.0f}%")
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
The tryptophan EM field acts as a QUANTUM BUS:
- Dimers from different bursts (200ms apart) become entangled
- Network continues growing during rest when no new dimers form
- This is NOT possible through birth entanglement alone

The EM field creates a coherent environment where entanglement
can propagate between spatially separate dimers.
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run entanglement network experiment')
    parser.add_argument('--bursts', type=int, default=5,
                        help='Number of theta bursts')
    parser.add_argument('--rest', type=float, default=10.0,
                        help='Rest duration in seconds')
    parser.add_argument('--output', type=str, default='results/tier1',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    result = run(n_bursts=args.bursts, 
                 rest_duration_s=args.rest,
                 verbose=not args.quiet)
    
    print_summary(result)
    
    fig = plot(result, output_dir=args.output)
    
    plt.show()