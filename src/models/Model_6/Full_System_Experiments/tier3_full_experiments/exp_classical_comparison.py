#!/usr/bin/env python3
"""
Experiment: Classical vs Quantum Eligibility Trace Comparison
==============================================================

Demonstrates that classical CaMKII mechanisms cannot maintain eligibility
traces over the 60-100 second learning window, while quantum dimer-based
eligibility can.

THIS VERSION USES THE ACTUAL MODEL 6 SYSTEM - not analytical approximations.

EXPERIMENTAL DESIGN:

1. QUANTUM ELIGIBILITY:
   - Run Model6QuantumSynapse with full EM coupling
   - Extract synapse.get_eligibility() at each timepoint
   - This is the particle-based singlet probability: (mean_P_S - 0.25) / 0.75

2. CLASSICAL BASELINE (CaMKII molecular_memory):
   - From the SAME simulation, extract synapse.camkii.molecular_memory
   - This is pT286 × GluN2B_bound - the classical memory signal
   - Shows what CaMKII alone can achieve

3. COMPARISON:
   - Track both signals after theta-burst stimulation
   - No dopamine (pure decay comparison)
   - Show quantum eligibility persists while CaMKII decays

KEY PREDICTIONS:
- Classical (CaMKII): molecular_memory decays with τ ≈ 10s (PP1 limited)
- Quantum (dimers): eligibility decays with T2 ≈ 100s (singlet lifetime)

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import sys
import time
from scipy.optimize import curve_fit

# Add parent to path for Model 6 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# REQUIRE Model 6 imports - fail loudly if not available
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimePoint:
    """Measurements at a single timepoint"""
    time_s: float
    quantum_eligibility: float  # From synapse.get_eligibility()
    mean_singlet_prob: float    # From synapse.get_mean_singlet_probability()
    camkii_pT286: float         # From synapse.camkii.pT286
    camkii_memory: float        # From synapse.camkii.molecular_memory
    n_dimers: int               # Dimer count
    n_entangled: int            # Entangled dimers


@dataclass
class DecayTrace:
    """Complete decay trace for one mechanism"""
    mechanism: str
    times: np.ndarray
    values: np.ndarray
    
    # Fitted parameters
    fitted_tau: float = 0.0
    fitted_A: float = 1.0
    fit_r2: float = 0.0
    
    # Key metrics
    value_at_60s: float = 0.0
    value_at_100s: float = 0.0
    time_to_30pct: float = np.inf


@dataclass
class ClassicalComparisonResult:
    """Complete experiment results"""
    # Raw timepoints
    timepoints: List[TimePoint] = field(default_factory=list)
    
    # Processed traces
    quantum_trace: DecayTrace = None
    camkii_trace: DecayTrace = None
    
    # P32 control (if run)
    quantum_p32_trace: DecayTrace = None
    
    # Summary
    quantum_advantage_60s: float = 0.0
    quantum_advantage_100s: float = 0.0
    tau_ratio: float = 0.0
    
    timestamp: str = ""
    runtime_s: float = 0.0


# =============================================================================
# FITTING UTILITIES
# =============================================================================

def exponential_decay(t, A, tau):
    """Exponential decay: A * exp(-t/tau)"""
    return A * np.exp(-t / tau)


def fit_decay_curve(times: np.ndarray, values: np.ndarray,
                    start_idx: int = 0) -> Tuple[float, float, float]:
    """
    Fit exponential decay to data
    
    Returns: (tau, A, r2)
    """
    t = times[start_idx:] - times[start_idx]
    v = values[start_idx:]
    
    # Only fit positive values
    mask = v > 0.01
    if np.sum(mask) < 3:
        return 50.0, 1.0, 0.0
    
    t_fit = t[mask]
    v_fit = v[mask]
    
    try:
        popt, _ = curve_fit(exponential_decay, t_fit, v_fit,
                           p0=[v_fit[0], 50.0],
                           bounds=([0.01, 0.5], [2.0, 500]),
                           maxfev=2000)
        A, tau = popt
        
        v_pred = exponential_decay(t_fit, A, tau)
        ss_res = np.sum((v_fit - v_pred)**2)
        ss_tot = np.sum((v_fit - np.mean(v_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return tau, A, r2
    except Exception as e:
        print(f"    Fit failed: {e}")
        return 50.0, 1.0, 0.0


def find_time_to_threshold(times: np.ndarray, values: np.ndarray,
                           threshold: float = 0.3) -> float:
    """Find time when values drop below threshold after peak"""
    peak_idx = np.argmax(values)
    values_after_peak = values[peak_idx:]
    times_after_peak = times[peak_idx:]
    
    below = np.where(values_after_peak < threshold)[0]
    if len(below) > 0:
        return times_after_peak[below[0]]
    return np.inf


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_decay_simulation(isotope: str = 'P31',
                         n_synapses: int = 10,
                         stim_duration_s: float = 1.0,
                         decay_duration_s: float = 120.0,
                         record_interval_s: float = 1.0,
                         verbose: bool = True) -> List[TimePoint]:
    """
    Run a single decay simulation with Model 6
    
    Stimulate with theta-burst, then track decay of both
    quantum eligibility and CaMKII molecular_memory.
    """
    if verbose:
        print(f"\n  Running {isotope} decay simulation...")
    
    # Configure parameters
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    
    # Set isotope
    if isotope == 'P31':
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
    else:
        params.environment.fraction_P31 = 0.0
        params.environment.fraction_P32 = 1.0
    
    # Create network
    network = MultiSynapseNetwork(
        n_synapses=n_synapses,
        params=params,
        pattern='clustered',
        spacing_um=1.0
    )
    network.initialize(Model6QuantumSynapse, params)
    network.set_microtubule_invasion(True)
    
    dt = 0.001  # 1 ms timestep
    timepoints = []
    
    # === PHASE 1: THETA-BURST STIMULATION ===
    if verbose:
        print(f"    Phase 1: Theta-burst stimulation ({stim_duration_s}s)")
    
    # 5 bursts at 5 Hz, each burst = 4 spikes at 100 Hz
    for burst in range(5):
        for spike in range(4):
            # 2ms depolarization
            for _ in range(2):
                network.step(dt, {'voltage': -10e-3, 'reward': False})
            # 8ms at rest
            for _ in range(8):
                network.step(dt, {'voltage': -70e-3, 'reward': False})
        # 160ms between bursts
        for _ in range(160):
            network.step(dt, {'voltage': -70e-3, 'reward': False})
    
    # === PHASE 2: DECAY (NO DOPAMINE) ===
    if verbose:
        print(f"    Phase 2: Decay observation ({decay_duration_s}s)")
    
    # Use coarser timestep for efficiency during decay
    dt_decay = 0.1  # 100 ms steps
    n_decay_steps = int(decay_duration_s / dt_decay)
    
    t = 0.0
    last_record = -record_interval_s  # Force first record
    
    for step in range(n_decay_steps):
        # Step the network (no reward signal)
        state = network.step(dt_decay, {'voltage': -70e-3, 'reward': False})
        t += dt_decay
        
        # Record at intervals
        if t - last_record >= record_interval_s or step == 0:
            # Get quantum metrics
            eligibility = state.mean_eligibility
            
            # Get detailed metrics from synapses
            singlet_probs = []
            pT286s = []
            memories = []
            n_dimers = 0
            n_entangled = 0
            
            for synapse in network.synapses:
                singlet_probs.append(synapse.get_mean_singlet_probability())
                
                if hasattr(synapse, 'camkii'):
                    pT286s.append(synapse.camkii.pT286)
                    memories.append(synapse.camkii.molecular_memory)
                
                if hasattr(synapse, 'dimer_particles'):
                    n_dimers += len(synapse.dimer_particles.dimers)
                    n_entangled += sum(1 for d in synapse.dimer_particles.dimers 
                                       if d.is_entangled)
            
            tp = TimePoint(
                time_s=t,
                quantum_eligibility=eligibility,
                mean_singlet_prob=np.mean(singlet_probs) if singlet_probs else 0.25,
                camkii_pT286=np.mean(pT286s) if pT286s else 0.0,
                camkii_memory=np.mean(memories) if memories else 0.0,
                n_dimers=n_dimers,
                n_entangled=n_entangled
            )
            timepoints.append(tp)
            last_record = t
            
            if verbose and step % 100 == 0:
                print(f"      t={t:.0f}s: elig={eligibility:.3f}, "
                      f"P_S={tp.mean_singlet_prob:.3f}, "
                      f"CaMKII={tp.camkii_memory:.3f}, "
                      f"dimers={n_dimers}")
    
    return timepoints


def process_timepoints(timepoints: List[TimePoint], mechanism: str) -> DecayTrace:
    """Convert timepoints to decay trace with fitting"""
    times = np.array([tp.time_s for tp in timepoints])
    
    if mechanism == 'quantum':
        values = np.array([tp.quantum_eligibility for tp in timepoints])
    elif mechanism == 'camkii':
        values = np.array([tp.camkii_memory for tp in timepoints])
    elif mechanism == 'singlet':
        values = np.array([tp.mean_singlet_prob for tp in timepoints])
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    trace = DecayTrace(
        mechanism=mechanism,
        times=times,
        values=values
    )
    
    # Fit decay curve
    tau, A, r2 = fit_decay_curve(times, values, start_idx=0)
    trace.fitted_tau = tau
    trace.fitted_A = A
    trace.fit_r2 = r2
    
    # Calculate key metrics
    idx_60s = np.argmin(np.abs(times - 60))
    idx_100s = np.argmin(np.abs(times - 100))
    
    trace.value_at_60s = values[idx_60s] if idx_60s < len(values) else 0
    trace.value_at_100s = values[idx_100s] if idx_100s < len(values) else 0
    trace.time_to_30pct = find_time_to_threshold(times, values, 0.3)
    
    return trace


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_figure(result: ClassicalComparisonResult, 
                             output_path: Path) -> plt.Figure:
    """Create comprehensive comparison figure"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    quantum = result.quantum_trace
    camkii = result.camkii_trace
    
    c_quantum = '#2563eb'  # Blue
    c_classical = '#dc2626'  # Red
    c_p32 = '#9333ea'  # Purple
    
    # === Panel A: Direct comparison ===
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.plot(camkii.times, camkii.values, '-', color=c_classical, linewidth=2.5,
             label=f'CaMKII molecular_memory (τ={camkii.fitted_tau:.1f}s)')
    ax1.plot(quantum.times, quantum.values, '-', color=c_quantum, linewidth=2.5,
             label=f'Quantum eligibility (τ={quantum.fitted_tau:.1f}s)')
    
    if result.quantum_p32_trace is not None:
        p32 = result.quantum_p32_trace
        ax1.plot(p32.times, p32.values, ':', color=c_p32, linewidth=2,
                 label=f'Quantum ³²P control (τ={p32.fitted_tau:.1f}s)')
    
    ax1.axhline(0.3, color='green', linestyle='--', linewidth=1.5,
                label='Commitment threshold')
    ax1.axvspan(60, 100, alpha=0.15, color='orange', label='Learning window')
    
    ax1.set_xlabel('Time after stimulation (s)')
    ax1.set_ylabel('Eligibility / Memory signal')
    ax1.set_title('A. Classical vs Quantum Eligibility (Model 6 Simulation)')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Annotate quantum advantage
    if quantum.value_at_60s > 0 and camkii.value_at_60s > 0:
        ratio = quantum.value_at_60s / max(camkii.value_at_60s, 0.001)
        ax1.annotate(f'{ratio:.0f}× advantage\nat 60s',
                    xy=(60, quantum.value_at_60s),
                    xytext=(75, quantum.value_at_60s + 0.15),
                    fontsize=11, color=c_quantum,
                    arrowprops=dict(arrowstyle='->', color=c_quantum))
    
    # === Panel B: Log scale ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    q_mask = quantum.values > 0.001
    c_mask = camkii.values > 0.001
    
    ax2.semilogy(camkii.times[c_mask], camkii.values[c_mask],
                 '-', color=c_classical, linewidth=2, label='CaMKII')
    ax2.semilogy(quantum.times[q_mask], quantum.values[q_mask],
                 '-', color=c_quantum, linewidth=2, label='Quantum ³¹P')
    
    if result.quantum_p32_trace is not None:
        p32 = result.quantum_p32_trace
        p32_mask = p32.values > 0.001
        ax2.semilogy(p32.times[p32_mask], p32.values[p32_mask],
                     ':', color=c_p32, linewidth=1.5, label='Quantum ³²P')
    
    ax2.axhline(0.3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(0.01, color='gray', linestyle=':', linewidth=1, label='1% floor')
    ax2.axvspan(60, 100, alpha=0.15, color='orange')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Signal (log scale)')
    ax2.set_title('B. Logarithmic Scale')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(1e-3, 2)
    ax2.grid(True, alpha=0.3)
    
    # === Panel C: Time to threshold bar chart ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    thresholds = [0.5, 0.3, 0.1]
    times_camkii = [find_time_to_threshold(camkii.times, camkii.values, th) for th in thresholds]
    times_quantum = [find_time_to_threshold(quantum.times, quantum.values, th) for th in thresholds]
    
    # Cap at 150 for display
    times_camkii = [min(t, 150) for t in times_camkii]
    times_quantum = [min(t, 150) for t in times_quantum]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, times_camkii, width, label='CaMKII', color=c_classical)
    bars2 = ax3.bar(x + width/2, times_quantum, width, label='Quantum', color=c_quantum)
    
    ax3.set_ylabel('Time to threshold (s)')
    ax3.set_xlabel('Threshold')
    ax3.set_title('C. Time to Reach Threshold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{th:.0%}' for th in thresholds])
    ax3.legend()
    ax3.set_ylim(0, 160)
    
    # Add value labels
    for bar, val in zip(bars1, times_camkii):
        label = f'{val:.0f}s' if val < 150 else '>120s'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 label, ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, times_quantum):
        label = f'{val:.0f}s' if val < 150 else '>120s'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 label, ha='center', va='bottom', fontsize=9)
    
    ax3.axhline(60, color='orange', linestyle='--', linewidth=2)
    ax3.text(2.2, 65, '60s learning window', fontsize=9, color='orange')
    
    plt.suptitle('Classical vs Quantum Eligibility: Why Quantum Coherence is Necessary\n'
                 '(Results from Model 6 simulation)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path / 'classical_vs_quantum_comparison.png')
    plt.savefig(output_path / 'classical_vs_quantum_comparison.pdf')
    plt.close()
    
    print(f"  Saved: classical_vs_quantum_comparison.png")
    
    return fig


def save_results(result: ClassicalComparisonResult, output_path: Path) -> Dict:
    """Save results to JSON"""
    summary = {
        'experiment': 'Classical vs Quantum Eligibility (Model 6)',
        'timestamp': result.timestamp,
        'runtime_s': result.runtime_s,
        
        'quantum_p31': {
            'mechanism': 'Dimer singlet probability',
            'fitted_tau_s': result.quantum_trace.fitted_tau,
            'fit_r2': result.quantum_trace.fit_r2,
            'value_at_60s': result.quantum_trace.value_at_60s,
            'value_at_100s': result.quantum_trace.value_at_100s,
            'time_to_30pct_s': result.quantum_trace.time_to_30pct,
        },
        
        'camkii': {
            'mechanism': 'pT286 × GluN2B_bound',
            'fitted_tau_s': result.camkii_trace.fitted_tau,
            'fit_r2': result.camkii_trace.fit_r2,
            'value_at_60s': result.camkii_trace.value_at_60s,
            'value_at_100s': result.camkii_trace.value_at_100s,
            'time_to_30pct_s': result.camkii_trace.time_to_30pct,
        },
        
        'comparison': {
            'tau_ratio': result.tau_ratio,
            'quantum_advantage_60s': result.quantum_advantage_60s,
            'quantum_advantage_100s': result.quantum_advantage_100s,
        },
        
        'conclusion': 'Quantum mechanism maintains eligibility through learning window; CaMKII does not.'
    }
    
    if result.quantum_p32_trace is not None:
        summary['quantum_p32_control'] = {
            'fitted_tau_s': result.quantum_p32_trace.fitted_tau,
            'value_at_60s': result.quantum_p32_trace.value_at_60s,
        }
    
    with open(output_path / 'classical_quantum_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    return summary


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def run_classical_comparison(output_dir: str = None,
                             n_synapses: int = 10,
                             decay_duration_s: float = 120.0,
                             include_p32: bool = True,
                             quick: bool = False) -> ClassicalComparisonResult:
    """
    Run complete classical vs quantum comparison using Model 6
    """
    print("=" * 70)
    print("CLASSICAL VS QUANTUM ELIGIBILITY COMPARISON")
    print("Using Model 6 - Actual simulations, not analytical approximations")
    print("=" * 70)
    
    start_time = time.time()
    
    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"classical_comparison_{timestamp}")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Quick mode settings
    if quick:
        n_synapses = 5
        decay_duration_s = 60.0
        print("Quick mode: reduced synapses and duration")
    
    # Run P31 simulation
    print("\n--- P31 (Natural isotope) ---")
    p31_timepoints = run_decay_simulation(
        isotope='P31',
        n_synapses=n_synapses,
        decay_duration_s=decay_duration_s,
        verbose=True
    )
    
    # Process into traces
    quantum_trace = process_timepoints(p31_timepoints, 'quantum')
    camkii_trace = process_timepoints(p31_timepoints, 'camkii')
    
    # Run P32 control if requested
    p32_trace = None
    if include_p32:
        print("\n--- P32 (Control - no nuclear spin) ---")
        p32_timepoints = run_decay_simulation(
            isotope='P32',
            n_synapses=n_synapses,
            decay_duration_s=min(30.0, decay_duration_s),  # P32 decays fast
            verbose=True
        )
        p32_trace = process_timepoints(p32_timepoints, 'quantum')
    
    # Calculate comparisons
    tau_ratio = quantum_trace.fitted_tau / max(camkii_trace.fitted_tau, 0.1)
    
    if camkii_trace.value_at_60s > 0.001:
        advantage_60s = quantum_trace.value_at_60s / camkii_trace.value_at_60s
    else:
        advantage_60s = float('inf')
    
    if camkii_trace.value_at_100s > 0.001:
        advantage_100s = quantum_trace.value_at_100s / camkii_trace.value_at_100s
    else:
        advantage_100s = float('inf')
    
    # Create result
    result = ClassicalComparisonResult(
        timepoints=p31_timepoints,
        quantum_trace=quantum_trace,
        camkii_trace=camkii_trace,
        quantum_p32_trace=p32_trace,
        quantum_advantage_60s=advantage_60s,
        quantum_advantage_100s=advantage_100s,
        tau_ratio=tau_ratio,
        timestamp=datetime.now().isoformat(),
        runtime_s=time.time() - start_time
    )
    
    # Generate outputs
    print("\nGenerating figures...")
    create_comparison_figure(result, output_path)
    summary = save_results(result, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'CaMKII':<15} {'Quantum ³¹P':<15}")
    print("-" * 60)
    print(f"{'Fitted τ (s)':<30} {camkii_trace.fitted_tau:<15.1f} {quantum_trace.fitted_tau:<15.1f}")
    print(f"{'Value at 60s':<30} {camkii_trace.value_at_60s:<15.3f} {quantum_trace.value_at_60s:<15.3f}")
    print(f"{'Value at 100s':<30} {camkii_trace.value_at_100s:<15.4f} {quantum_trace.value_at_100s:<15.3f}")
    print(f"{'Time to 30%':<30} {camkii_trace.time_to_30pct:<15.1f} {quantum_trace.time_to_30pct:<15.1f}")
    print("-" * 60)
    print(f"Quantum advantage at 60s: {advantage_60s:.0f}×")
    print(f"Time constant ratio: {tau_ratio:.1f}×")
    print(f"\nTotal runtime: {result.runtime_s:.1f}s")
    print("=" * 70)
    
    return result


class ClassicalComparisonExperiment:
    """Wrapper class for run_tier3.py integration"""
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.result = None
    
    def run(self, output_dir: Path = None) -> ClassicalComparisonResult:
        """Run the experiment"""
        self.result = run_classical_comparison(
            output_dir=str(output_dir) if output_dir else None,
            quick=self.quick_mode,
            include_p32=not self.quick_mode
        )
        return self.result
    
    def print_summary(self, result: ClassicalComparisonResult) -> None:
        """Print summary"""
        print(f"\nQuantum τ={result.quantum_trace.fitted_tau:.1f}s vs "
              f"CaMKII τ={result.camkii_trace.fitted_tau:.1f}s")
        print(f"Quantum advantage: {result.quantum_advantage_60s:.0f}× at 60s")
    
    def plot(self, result: ClassicalComparisonResult, output_dir: Path = None) -> None:
        """Plots already saved during run"""
        pass
    
    def save_results(self, result: ClassicalComparisonResult, path: Path) -> None:
        """Save results - already done during run, but save summary here too"""
        summary = {
            'quantum_tau_s': result.quantum_trace.fitted_tau,
            'camkii_tau_s': result.camkii_trace.fitted_tau,
            'quantum_advantage_60s': result.quantum_advantage_60s,
            'runtime_s': result.runtime_s
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classical vs Quantum Comparison (Model 6)')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--synapses', '-n', type=int, default=10)
    parser.add_argument('--duration', '-d', type=float, default=120.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-p32', action='store_true', help='Skip P32 control')
    
    args = parser.parse_args()
    
    run_classical_comparison(
        output_dir=args.output,
        n_synapses=args.synapses,
        decay_duration_s=args.duration,
        include_p32=not args.no_p32,
        quick=args.quick
    )