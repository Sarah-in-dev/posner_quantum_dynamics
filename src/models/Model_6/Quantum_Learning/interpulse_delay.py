"""
interpulse_delay_v2.py - Corrected interpulse delay experiment

Measures plasticity at appropriate timescales (minutes, not seconds)
to capture DDSC-driven structural changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from model6_core import QuantumSynapseModel
from typing import List, Tuple

# Experimental parameters
P31_DELAYS = [5, 15, 30, 45, 60, 90]  # seconds
P32_DELAYS = [0.1, 0.3, 0.5, 1.0, 2.0]  # seconds (all should fail threshold)

# Measurement timepoints (post-plateau)
MEASUREMENT_TIMES = [30, 120, 300, 1200]  # 30s, 2min, 5min, 20min

# Protocol timing
STIM_DURATION = 0.2  # 200ms
PLATEAU_DURATION = 0.3  # 300ms


def run_single_trial(
    delay: float,
    measurement_time: float,
    isotope: str = 'P31',
    verbose: bool = False
) -> dict:
    """
    Run single trial of interpulse delay experiment.
    
    Protocol:
    1. Stimulation (200ms) - creates dimers, establishes eligibility
    2. Delay (variable) - eligibility decays
    3. Plateau (300ms) - checks eligibility, triggers DDSC if above threshold
    4. Wait (measurement_time) - DDSC drives structural changes
    5. Measure - record final plasticity state
    """
    # Initialize model with appropriate isotope
    T2 = 67.6 if isotope == 'P31' else 0.3
    model = QuantumSynapseModel(T2=T2)
    
    # Phase 1: Stimulation
    model.apply_stimulation(duration=STIM_DURATION)
    
    # Record initial state
    initial_strength = model.get_synaptic_strength()
    
    # Phase 2: Delay (eligibility decaying)
    model.run(duration=delay)
    eligibility_at_plateau = model.quantum_state.get_eligibility()
    
    # Phase 3: Plateau (check gate, trigger DDSC)
    gate_opened = model.apply_plateau(current_time=model.time)
    model.run(duration=PLATEAU_DURATION)
    
    # Phase 4: Wait for structural changes
    model.run(duration=measurement_time)
    
    # Phase 5: Measure
    final_strength = model.get_synaptic_strength()
    plasticity = (final_strength - initial_strength) / initial_strength * 100
    
    # Collect results
    result = {
        'delay': delay,
        'measurement_time': measurement_time,
        'isotope': isotope,
        'eligibility_at_plateau': eligibility_at_plateau,
        'gate_opened': gate_opened,
        'initial_strength': initial_strength,
        'final_strength': final_strength,
        'plasticity_percent': plasticity,
        'ddsc_state': model.ddsc.get_state()
    }
    
    if verbose:
        print(f"\n=== Trial: {isotope}, delay={delay}s, measure={measurement_time}s ===")
        print(f"  Eligibility at plateau: {eligibility_at_plateau:.3f}")
        print(f"  Gate opened: {gate_opened}")
        print(f"  Integrated DDSC: {result['ddsc_state']['integrated_ddsc']:.2f}")
        print(f"  Structural drive: {result['ddsc_state']['structural_drive']:.3f}")
        print(f"  Plasticity: {plasticity:+.1f}%")
    
    return result


def run_timeseries_experiment(
    delays: List[float],
    measurement_times: List[float],
    isotope: str = 'P31',
    n_trials: int = 1
) -> List[dict]:
    """
    Run full experiment matrix: delays × measurement_times × trials
    """
    results = []
    
    for delay in delays:
        for mtime in measurement_times:
            for trial in range(n_trials):
                result = run_single_trial(
                    delay=delay,
                    measurement_time=mtime,
                    isotope=isotope,
                    verbose=(trial == 0)  # Print first trial only
                )
                result['trial'] = trial
                results.append(result)
    
    return results


def plot_results(p31_results: List[dict], p32_results: List[dict]):
    """
    Create publication-quality figure showing:
    A) Plasticity vs measurement time for different delays (P31)
    B) Final plasticity vs delay (comparing P31 vs P32)
    C) Eligibility decay curves for reference
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Plasticity accumulation over time (P31 only)
    ax = axes[0, 0]
    delays_to_plot = [5, 30, 60]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(delays_to_plot)))
    
    for delay, color in zip(delays_to_plot, colors):
        subset = [r for r in p31_results if r['delay'] == delay]
        times = [r['measurement_time'] / 60 for r in subset]  # Convert to minutes
        plasticity = [r['plasticity_percent'] for r in subset]
        ax.plot(times, plasticity, 'o-', color=color, label=f'delay={delay}s', linewidth=2, markersize=8)
    
    ax.set_xlabel('Measurement Time (minutes)', fontsize=12)
    ax.set_ylabel('Plasticity (%)', fontsize=12)
    ax.set_title('A. Plasticity Accumulation (³¹P)', fontsize=14)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 25)
    
    # Panel B: Final plasticity vs delay (P31 vs P32)
    ax = axes[0, 1]
    
    # P31 at 20 min measurement
    p31_final = [r for r in p31_results if r['measurement_time'] == 1200]
    p31_delays = [r['delay'] for r in p31_final]
    p31_plasticity = [r['plasticity_percent'] for r in p31_final]
    ax.plot(p31_delays, p31_plasticity, 'o-', color='blue', label='³¹P', linewidth=2, markersize=8)
    
    # P32 at 20 min measurement
    p32_final = [r for r in p32_results if r['measurement_time'] == 1200]
    p32_delays = [r['delay'] for r in p32_final]
    p32_plasticity = [r['plasticity_percent'] for r in p32_final]
    ax.plot(p32_delays, p32_plasticity, 's--', color='red', label='³²P', linewidth=2, markersize=8)
    
    ax.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax.set_ylabel('Plasticity at 20 min (%)', fontsize=12)
    ax.set_title('B. Isotope Comparison', fontsize=14)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel C: Eligibility at plateau vs delay
    ax = axes[1, 0]
    
    p31_elig = [r['eligibility_at_plateau'] for r in p31_final]
    ax.plot(p31_delays, p31_elig, 'o-', color='blue', label='³¹P', linewidth=2, markersize=8)
    
    p32_elig = [r['eligibility_at_plateau'] for r in p32_final]
    ax.plot(p32_delays, p32_elig, 's--', color='red', label='³²P', linewidth=2, markersize=8)
    
    ax.axhline(y=0.3, color='green', linestyle=':', label='Threshold', linewidth=2)
    ax.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax.set_ylabel('Eligibility at Plateau', fontsize=12)
    ax.set_title('C. Eligibility Decay', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Panel D: DDSC profiles for different eligibility levels
    ax = axes[1, 1]
    
    # Theoretical DDSC curves
    t = np.linspace(0, 300, 1000)
    tau_rise, tau_decay = 15.0, 50.0
    
    for elig, color, label in [(0.9, 'darkgreen', 'High elig (0.9)'),
                                (0.5, 'orange', 'Med elig (0.5)'),
                                (0.25, 'gray', 'Low elig (0.25)')]:
        ddsc = elig * (1 - np.exp(-t/tau_rise)) * np.exp(-t/tau_decay)
        ax.plot(t/60, ddsc, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Time After Plateau (minutes)', fontsize=12)
    ax.set_ylabel('DDSC Activation', fontsize=12)
    ax.set_title('D. DDSC Dynamics', fontsize=14)
    ax.legend()
    ax.axvline(x=35/60, color='gray', linestyle=':', alpha=0.5)
    ax.text(35/60 + 0.1, 0.5, 'Peak ~35s', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('interpulse_delay_results_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    print("=" * 60)
    print("INTERPULSE DELAY EXPERIMENT v2")
    print("Corrected timescales with DDSC integration")
    print("=" * 60)
    
    # Run P31 experiments
    print("\n>>> Running ³¹P experiments...")
    p31_results = run_timeseries_experiment(
        delays=P31_DELAYS,
        measurement_times=MEASUREMENT_TIMES,
        isotope='P31',
        n_trials=1
    )
    
    # Run P32 experiments
    print("\n>>> Running ³²P experiments...")
    p32_results = run_timeseries_experiment(
        delays=P32_DELAYS,
        measurement_times=MEASUREMENT_TIMES,
        isotope='P32',
        n_trials=1
    )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Final Plasticity at 20 minutes")
    print("=" * 60)
    
    print("\n³¹P Results:")
    print(f"{'Delay (s)':<12} {'Eligibility':<12} {'Gate':<8} {'Plasticity':<12}")
    print("-" * 44)
    for r in [r for r in p31_results if r['measurement_time'] == 1200]:
        print(f"{r['delay']:<12} {r['eligibility_at_plateau']:<12.3f} {str(r['gate_opened']):<8} {r['plasticity_percent']:+.1f}%")
    
    print("\n³²P Results:")
    print(f"{'Delay (s)':<12} {'Eligibility':<12} {'Gate':<8} {'Plasticity':<12}")
    print("-" * 44)
    for r in [r for r in p32_results if r['measurement_time'] == 1200]:
        print(f"{r['delay']:<12} {r['eligibility_at_plateau']:<12.3f} {str(r['gate_opened']):<8} {r['plasticity_percent']:+.1f}%")
    
    # Generate plots
    print("\n>>> Generating figures...")
    plot_results(p31_results, p32_results)
    
    print("\n>>> Done!")


if __name__ == "__main__":
    main()