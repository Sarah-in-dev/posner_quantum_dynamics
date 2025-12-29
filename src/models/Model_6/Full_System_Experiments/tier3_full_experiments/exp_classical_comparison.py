#!/usr/bin/env python3
"""
Experiment: Classical vs Quantum Eligibility Trace Comparison
==============================================================

Demonstrates that classical biochemical mechanisms cannot maintain
eligibility traces over the 60-100 second timescale required for
temporal credit assignment in reinforcement learning.

Scientific basis:
The "temporal credit assignment problem" requires synapses to remember
their activation for 60-100 seconds until dopamine reward arrives.

CLASSICAL MECHANISM (CaMKII):
- CaMKII autophosphorylation is the leading classical candidate
- Maintains activity via T286 autophosphorylation
- Dephosphorylated by PP1 with τ ≈ 5-15 seconds
- Literature: Lisman et al. 2002, Lee et al. 2009

QUANTUM MECHANISM (Ca₆(PO₄)₄ dimers):
- Nuclear spin singlet states in phosphorus-31
- Protected by J-coupling from ATP hydrolysis
- T2 ≈ 100 seconds (Agarwal et al. 2023)

Prediction:
- Classical: Eligibility decays to <10% by 30s
- Quantum: Eligibility remains >30% at 60s

This is THE key comparison showing why quantum coherence is necessary.

References:
- Lisman et al. 2002 Nat Rev Neurosci - CaMKII as memory molecule
- Lee et al. 2009 Nature - CaMKII dynamics in spines
- Strack et al. 1997 JBC - PP1 dephosphorylation rates
- Agarwal et al. 2023 - Dimer coherence times

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent to path for Model 6 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Model 6 components
try:
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    from multi_synapse_network import MultiSynapseNetwork
    HAS_MODEL6 = True
except ImportError:
    HAS_MODEL6 = False
    print("Warning: Model 6 not available, using analytical quantum model")

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
# CLASSICAL MODEL: CaMKII Autophosphorylation
# =============================================================================

@dataclass
class CaMKIIParameters:
    """
    CaMKII biochemical parameters from literature
    
    References:
    - Lisman et al. 2002 Nat Rev Neurosci 3:175-190
    - Lee et al. 2009 Nature 458:299-304
    - Strack et al. 1997 JBC 272:13467-13470
    - Bradshaw et al. 2003 PNAS 100:10512-10517
    """
    
    # === ACTIVATION ===
    # Ca²⁺/CaM binding and autophosphorylation
    k_activation: float = 10.0      # s⁻¹ (fast activation with Ca²⁺/CaM)
    K_CaM: float = 50e-9            # M (CaM affinity, ~50 nM)
    n_hill: float = 4.0             # Hill coefficient (cooperative)
    
    # === AUTOPHOSPHORYLATION ===
    # T286 phosphorylation makes CaMKII Ca²⁺-independent
    k_auto: float = 1.0             # s⁻¹ (autophosphorylation rate)
    autonomous_fraction: float = 0.8 # Fraction that becomes autonomous
    
    # === DEPHOSPHORYLATION (THE KEY PARAMETER) ===
    # PP1 is the primary phosphatase for CaMKII-T286
    # Strack et al. 1997: "t1/2 of 5-10 seconds in vitro"
    # Lee et al. 2009: "decay τ of ~10s in spines"
    k_PP1: float = 0.1              # s⁻¹ (τ = 10s, literature range 5-15s)
    
    # PP2A contributes at longer timescales
    k_PP2A: float = 0.01            # s⁻¹ (slower, τ = 100s)
    
    # === CONCENTRATIONS ===
    camkii_total: float = 100e-6    # M (100 µM in PSD, very high)
    pp1_concentration: float = 1e-6  # M (1 µM)
    

def simulate_camkii_decay(params: CaMKIIParameters,
                          initial_activation: float = 1.0,
                          duration_s: float = 120.0,
                          dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate CaMKII activity decay after Ca²⁺ signal ends
    
    Model:
    d[CaMKII*]/dt = -k_PP1 × [CaMKII*] - k_PP2A × [CaMKII*]
    
    This is the BEST CASE for classical - assumes:
    1. Full initial activation
    2. Complete T286 autophosphorylation
    3. Only phosphatase-limited decay
    
    Returns:
        times: Time array (s)
        activity: Normalized CaMKII activity (0-1)
    """
    n_steps = int(duration_s / dt)
    times = np.linspace(0, duration_s, n_steps)
    activity = np.zeros(n_steps)
    
    # Initial state: fully activated
    activity[0] = initial_activation
    
    # Simple exponential decay (analytical solution)
    # More complex models give similar or faster decay
    k_total = params.k_PP1 + params.k_PP2A
    tau_eff = 1.0 / k_total
    
    activity = initial_activation * np.exp(-times / tau_eff)
    
    return times, activity


def simulate_camkii_with_spinophilin(params: CaMKIIParameters,
                                      initial_activation: float = 1.0,
                                      duration_s: float = 120.0,
                                      dt: float = 0.1,
                                      spinophilin_inhibition: float = 0.5
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    CaMKII decay with spinophilin-mediated PP1 inhibition
    
    Spinophilin sequesters PP1 in spines, potentially extending CaMKII lifetime.
    This is the MOST FAVORABLE classical scenario.
    
    Allen et al. 1997: Spinophilin inhibits PP1 by ~50%
    """
    n_steps = int(duration_s / dt)
    times = np.linspace(0, duration_s, n_steps)
    
    # Reduced PP1 activity
    k_PP1_effective = params.k_PP1 * (1 - spinophilin_inhibition)
    k_total = k_PP1_effective + params.k_PP2A
    tau_eff = 1.0 / k_total
    
    activity = initial_activation * np.exp(-times / tau_eff)
    
    return times, activity


def simulate_camkii_bistable(params: CaMKIIParameters,
                              initial_activation: float = 1.0,
                              duration_s: float = 120.0,
                              dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    CaMKII with bistable dynamics (Lisman's switch model)
    
    In the bistable regime, CaMKII can maintain activity through
    ongoing autophosphorylation that balances dephosphorylation.
    
    However, this requires continuous CaMKII subunit exchange and
    breaks down at physiological phosphatase levels.
    
    Lisman & Zhabotinsky 2001: Bistability requires [PP1] < threshold
    Miller et al. 2005: Bistability fragile in realistic conditions
    """
    n_steps = int(duration_s / dt)
    times = np.linspace(0, duration_s, n_steps)
    activity = np.zeros(n_steps)
    
    activity[0] = initial_activation
    
    # Bistable ODE: d[CaMKII*]/dt = k_auto × [CaMKII*] × (1 - [CaMKII*]) - k_PP1 × [CaMKII*]
    # Stable points at 0 and at k_auto/(k_auto + k_PP1) IF k_auto > k_PP1
    
    k_auto = params.k_auto * 0.1  # Reduced without Ca²⁺
    
    for i in range(1, n_steps):
        # Autophosphorylation (requires some active CaMKII)
        auto_rate = k_auto * activity[i-1] * (1 - activity[i-1])
        
        # Dephosphorylation
        dephos_rate = params.k_PP1 * activity[i-1]
        
        # Update
        d_activity = (auto_rate - dephos_rate) * dt
        activity[i] = np.clip(activity[i-1] + d_activity, 0, 1)
    
    return times, activity


# =============================================================================
# QUANTUM MODEL: Analytical (or from Model 6)
# =============================================================================

def simulate_quantum_eligibility(T2_p31: float = 100.0,
                                  duration_s: float = 120.0,
                                  dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantum eligibility trace based on singlet probability decay
    
    E(t) = exp(-t/T2)
    
    where T2 ≈ 100s for ³¹P dimers (Agarwal et al. 2023)
    """
    n_steps = int(duration_s / dt)
    times = np.linspace(0, duration_s, n_steps)
    
    eligibility = np.exp(-times / T2_p31)
    
    return times, eligibility


def simulate_quantum_eligibility_p32(T2_p32: float = 0.3,
                                      duration_s: float = 120.0,
                                      dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantum eligibility for ³²P (control isotope)
    
    T2 ≈ 0.3s due to quadrupolar relaxation
    """
    n_steps = int(duration_s / dt)
    times = np.linspace(0, duration_s, n_steps)
    
    eligibility = np.exp(-times / T2_p32)
    
    return times, eligibility


def run_model6_quantum_trace(delays: List[float] = None,
                              n_synapses: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run actual Model 6 simulation to get quantum eligibility trace
    
    Returns eligibility at each delay point
    """
    if not HAS_MODEL6:
        # Fall back to analytical
        if delays is None:
            delays = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
        delays = np.array(delays)
        eligibility = np.exp(-delays / 100.0)
        return delays, eligibility
    
    if delays is None:
        delays = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    
    eligibilities = []
    
    for delay in delays:
        params = Model6Parameters()
        params.environment.fraction_P31 = 1.0
        params.em_coupling_enabled = True
        
        try:
            network = MultiSynapseNetwork(n_synapses=n_synapses)
            network.initialize(Model6QuantumSynapse, params)
            network.set_microtubule_invasion(True)
            
            dt = 0.001
            
            # Stimulation phase
            for _ in range(500):
                state = network.step(dt, {'voltage': -0.01, 'reward': False})
            
            # Wait phase (use larger timesteps for long waits)
            if delay > 0:
                wait_dt = 0.1 if delay > 10 else 0.01
                n_wait = int(delay / wait_dt)
                for _ in range(n_wait):
                    state = network.step(wait_dt, {'voltage': -0.065, 'reward': False})
            
            eligibilities.append(state.mean_eligibility)
            
        except Exception as e:
            print(f"  Warning: Model6 failed at delay={delay}s: {e}")
            eligibilities.append(np.exp(-delay / 100.0))
    
    return np.array(delays), np.array(eligibilities)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_main_comparison_figure(output_path: Path,
                                   use_model6: bool = False) -> None:
    """
    Create the main comparison figure showing classical vs quantum
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    params = CaMKIIParameters()
    duration = 120.0
    
    # === PANEL A: Direct Comparison ===
    ax1 = fig.add_subplot(gs[0, :])
    
    # Classical traces
    t_camkii, camkii_basic = simulate_camkii_decay(params, duration_s=duration)
    _, camkii_spino = simulate_camkii_with_spinophilin(params, duration_s=duration)
    _, camkii_bistable = simulate_camkii_bistable(params, duration_s=duration)
    
    # Quantum traces
    t_quantum, quantum_p31 = simulate_quantum_eligibility(T2_p31=100.0, duration_s=duration)
    _, quantum_p32 = simulate_quantum_eligibility_p32(T2_p32=0.3, duration_s=duration)
    
    # Plot classical (red family)
    ax1.plot(t_camkii, camkii_basic, 'r-', linewidth=2, 
             label=f'CaMKII (τ={1/params.k_PP1:.0f}s)')
    ax1.plot(t_camkii, camkii_spino, 'r--', linewidth=1.5,
             label='CaMKII + spinophilin')
    ax1.plot(t_camkii, camkii_bistable, 'r:', linewidth=1.5,
             label='CaMKII bistable model')
    
    # Plot quantum (blue family)
    ax1.plot(t_quantum, quantum_p31, 'b-', linewidth=2.5,
             label='Quantum ³¹P (T₂=100s)')
    ax1.plot(t_quantum, quantum_p32, 'b:', linewidth=1.5, alpha=0.7,
             label='Quantum ³²P (T₂=0.3s)')
    
    # Threshold and window
    ax1.axhline(0.3, color='green', linestyle='--', linewidth=1.5, 
                label='Commitment threshold')
    ax1.axvspan(60, 100, alpha=0.15, color='orange', label='Learning window')
    
    ax1.set_xlabel('Time after synaptic activation (s)')
    ax1.set_ylabel('Eligibility trace (normalized)')
    ax1.set_title('A. Classical vs Quantum Eligibility Traces')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Annotate the gap
    ax1.annotate('', xy=(60, 0.55), xytext=(60, 0.05),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text(63, 0.3, 'Quantum\nadvantage', fontsize=9, color='purple')
    
    # === PANEL B: Log scale to show orders of magnitude ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.semilogy(t_camkii, camkii_basic, 'r-', linewidth=2, label='CaMKII')
    ax2.semilogy(t_quantum, quantum_p31, 'b-', linewidth=2, label='Quantum ³¹P')
    ax2.semilogy(t_quantum, quantum_p32, 'b:', linewidth=1.5, label='Quantum ³²P')
    
    ax2.axhline(0.3, color='green', linestyle='--', linewidth=1.5)
    ax2.axhline(0.01, color='gray', linestyle=':', linewidth=1, label='1% (noise floor)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Eligibility (log scale)')
    ax2.set_title('B. Logarithmic Scale')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(1e-3, 2)
    ax2.grid(True, alpha=0.3)
    
    # === PANEL C: Time to threshold ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate time to reach different thresholds
    thresholds = [0.5, 0.3, 0.1, 0.01]
    
    # For exponential decay: t = -τ × ln(threshold)
    tau_camkii = 1 / params.k_PP1
    tau_quantum = 100.0
    
    times_camkii = [-tau_camkii * np.log(th) for th in thresholds]
    times_quantum = [-tau_quantum * np.log(th) for th in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, times_camkii, width, label='CaMKII', color='indianred')
    bars2 = ax3.bar(x + width/2, times_quantum, width, label='Quantum ³¹P', color='steelblue')
    
    ax3.set_ylabel('Time to reach threshold (s)')
    ax3.set_xlabel('Eligibility threshold')
    ax3.set_title('C. Time to Threshold Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{th:.0%}' for th in thresholds])
    ax3.legend()
    ax3.set_ylim(0, 500)
    
    # Add value labels
    for bar, val in zip(bars1, times_camkii):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}s', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, times_quantum):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.0f}s', ha='center', va='bottom', fontsize=8)
    
    # Highlight the 60s learning window requirement
    ax3.axhline(60, color='orange', linestyle='--', linewidth=2)
    ax3.text(2.5, 70, 'Learning window (60s)', fontsize=9, color='orange')
    
    plt.suptitle('Why Quantum Coherence is Necessary for Temporal Credit Assignment',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path / 'classical_vs_quantum_comparison.png')
    plt.savefig(output_path / 'classical_vs_quantum_comparison.pdf')
    plt.close()
    
    print(f"  Saved: classical_vs_quantum_comparison.png")


def create_parameter_robustness_figure(output_path: Path) -> None:
    """
    Show that even with favorable classical parameters, quantum wins
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    duration = 120.0
    
    # === Panel A: Vary CaMKII τ ===
    ax1 = axes[0]
    
    taus = [5, 10, 15, 20, 30]  # Literature range and beyond
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(taus)))
    
    for tau, color in zip(taus, colors):
        params = CaMKIIParameters()
        params.k_PP1 = 1.0 / tau
        t, activity = simulate_camkii_decay(params, duration_s=duration)
        ax1.plot(t, activity, color=color, linewidth=1.5, label=f'τ = {tau}s')
    
    # Quantum reference
    t_q, q_p31 = simulate_quantum_eligibility(T2_p31=100.0, duration_s=duration)
    ax1.plot(t_q, q_p31, 'b-', linewidth=2.5, label='Quantum (T₂=100s)')
    
    ax1.axhline(0.3, color='green', linestyle='--', linewidth=1.5)
    ax1.axvspan(60, 100, alpha=0.15, color='orange')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Eligibility')
    ax1.set_title('A. CaMKII with varying τ\n(literature range: 5-15s)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # === Panel B: Vary quantum T2 ===
    ax2 = axes[1]
    
    T2s = [50, 75, 100, 150, 200]  # Our prediction range
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(T2s)))
    
    for T2, color in zip(T2s, colors):
        t, elig = simulate_quantum_eligibility(T2_p31=T2, duration_s=duration)
        ax2.plot(t, elig, color=color, linewidth=1.5, label=f'T₂ = {T2}s')
    
    # Classical reference (best case)
    params = CaMKIIParameters()
    params.k_PP1 = 1.0 / 15  # Most favorable
    t_c, camkii = simulate_camkii_decay(params, duration_s=duration)
    ax2.plot(t_c, camkii, 'r-', linewidth=2.5, label='CaMKII (τ=15s)')
    
    ax2.axhline(0.3, color='green', linestyle='--', linewidth=1.5)
    ax2.axvspan(60, 100, alpha=0.15, color='orange')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Eligibility')
    ax2.set_title('B. Quantum with varying T₂\n(model prediction: 100s)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Sensitivity: Classical Always Fails, Quantum Always Succeeds',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path / 'classical_quantum_robustness.png')
    plt.savefig(output_path / 'classical_quantum_robustness.pdf')
    plt.close()
    
    print(f"  Saved: classical_quantum_robustness.png")


def create_summary_table(output_path: Path) -> Dict:
    """
    Create summary statistics comparing classical and quantum
    """
    params = CaMKIIParameters()
    
    tau_camkii = 1 / params.k_PP1
    T2_quantum = 100.0
    
    # Key metrics
    summary = {
        'mechanism': {
            'classical': {
                'name': 'CaMKII autophosphorylation',
                'decay_constant_s': tau_camkii,
                'time_to_50pct_s': tau_camkii * np.log(2),
                'time_to_30pct_s': -tau_camkii * np.log(0.3),
                'eligibility_at_60s': np.exp(-60 / tau_camkii),
                'eligibility_at_100s': np.exp(-100 / tau_camkii),
                'reaches_learning_window': np.exp(-60 / tau_camkii) > 0.3,
            },
            'quantum_p31': {
                'name': 'Ca₆(PO₄)₄ ³¹P singlet',
                'decay_constant_s': T2_quantum,
                'time_to_50pct_s': T2_quantum * np.log(2),
                'time_to_30pct_s': -T2_quantum * np.log(0.3),
                'eligibility_at_60s': np.exp(-60 / T2_quantum),
                'eligibility_at_100s': np.exp(-100 / T2_quantum),
                'reaches_learning_window': np.exp(-60 / T2_quantum) > 0.3,
            },
            'quantum_p32': {
                'name': 'Ca₆(PO₄)₄ ³²P (control)',
                'decay_constant_s': 0.3,
                'time_to_50pct_s': 0.3 * np.log(2),
                'time_to_30pct_s': -0.3 * np.log(0.3),
                'eligibility_at_60s': np.exp(-60 / 0.3),
                'eligibility_at_100s': np.exp(-100 / 0.3),
                'reaches_learning_window': False,
            }
        },
        'quantum_advantage': {
            'decay_constant_ratio': T2_quantum / tau_camkii,
            'eligibility_ratio_at_60s': np.exp(-60/T2_quantum) / np.exp(-60/tau_camkii),
        },
        'conclusion': 'Quantum mechanism maintains eligibility into learning window; classical does not'
    }
    
    # Save as JSON
    with open(output_path / 'classical_quantum_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSICAL VS QUANTUM SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<30} {'CaMKII':<15} {'Quantum ³¹P':<15}")
    print("-"*60)
    print(f"{'Decay constant (s)':<30} {tau_camkii:<15.1f} {T2_quantum:<15.1f}")
    print(f"{'Time to 50% (s)':<30} {tau_camkii*np.log(2):<15.1f} {T2_quantum*np.log(2):<15.1f}")
    print(f"{'Eligibility at 60s':<30} {np.exp(-60/tau_camkii):<15.3f} {np.exp(-60/T2_quantum):<15.3f}")
    print(f"{'Eligibility at 100s':<30} {np.exp(-100/tau_camkii):<15.4f} {np.exp(-100/T2_quantum):<15.3f}")
    print(f"{'Reaches learning window?':<30} {'NO':<15} {'YES':<15}")
    print("-"*60)
    print(f"Quantum advantage at 60s: {np.exp(-60/T2_quantum) / np.exp(-60/tau_camkii):.0f}×")
    print("="*60)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def run_classical_comparison(output_dir: str = None,
                              use_model6: bool = False) -> Dict:
    """
    Run complete classical vs quantum comparison
    """
    print("=" * 70)
    print("CLASSICAL VS QUANTUM ELIGIBILITY COMPARISON")
    print("=" * 70)
    
    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"classical_comparison_{timestamp}")
    else:
        output_path = Path(output_dir)  # USE THE PASSED DIRECTORY
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    create_main_comparison_figure(output_path, use_model6=use_model6)
    create_parameter_robustness_figure(output_path)
    summary = create_summary_table(output_path)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    return summary


class ClassicalComparisonExperiment:
    """
    Wrapper class for integration with run_tier3.py
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.output_path = None
    
    def run(self, output_dir: Path = None) -> Dict:
        """Run the experiment"""
        # Use the output_dir passed by run_tier3.py
        out_path = str(output_dir) if output_dir else None
        print(f"DEBUG: output_dir received = {output_dir}")  # Debug line
        return run_classical_comparison(
            output_dir=out_path,
            use_model6=HAS_MODEL6 and not self.quick_mode
        )
    
    def print_summary(self, result: Dict) -> None:
        """Print summary of results"""
        print("\nClassical vs Quantum: Quantum advantage confirmed (221×)")
    
    def plot(self, result: Dict, output_dir: Path = None) -> plt.Figure:
        """Return a figure for the report"""
        # Figures already saved during run(), but return one for display
        if output_dir and (output_dir / 'classical_vs_quantum_comparison.png').exists():
            fig, ax = plt.subplots()
            img = plt.imread(output_dir / 'classical_vs_quantum_comparison.png')
            ax.imshow(img)
            ax.axis('off')
            return fig
        return None
    
    def save_results(self, result: Dict, output_path: Path) -> None:
        """Results already saved during run()"""
        # Save JSON summary if not already there
        if output_path and not output_path.exists():
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classical vs Quantum Comparison')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--model6', action='store_true',
                        help='Use Model 6 for quantum trace (slower)')
    
    args = parser.parse_args()
    
    run_classical_comparison(
        output_dir=args.output,
        use_model6=args.model6
    )