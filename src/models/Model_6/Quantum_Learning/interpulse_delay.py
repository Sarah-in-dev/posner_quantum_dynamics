"""
interpulse_delay_proper.py - Interpulse Delay Experiment Using Model 6 As Designed

This experiment uses the ACTUAL Model 6 architecture:
- Model6Parameters sets isotope via fraction_P31
- Quantum coherence emerges from the physics
- Eligibility traces from dimer coherence
- DDSC gates structural plasticity
- Spine plasticity accumulates over minutes

NO SHORTCUTS - let the physics do the work.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Delays to test (seconds)
P31_DELAYS = [5, 15, 30, 45, 60, 90]  # Full range for P31
P32_DELAYS = [0.1, 0.3, 0.5, 1.0, 2.0]  # Short delays for P32

# Measurement timepoints after plateau (seconds)
MEASUREMENT_TIMES = [30, 120, 300, 600, 1200]  # 30s, 2min, 5min, 10min, 20min

# Quick test mode - shorter experiment
QUICK_TEST = True
if QUICK_TEST:
    P31_DELAYS = [5, 60]
    P32_DELAYS = [0.3, 1.0]
    MEASUREMENT_TIMES = [300]  # Just 5 minutes

# Protocol timing
STIM_VOLTAGE = -20e-3       # Depolarized (opens Ca channels)
REST_VOLTAGE = -70e-3       # Resting potential
STIM_DURATION = 0.2         # 200ms stimulation
PLATEAU_DURATION = 0.3      # 300ms plateau

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_model_params(isotope: str = 'P31') -> Model6Parameters:
    """
    Create Model6Parameters with appropriate isotope setting.
    
    Args:
        isotope: 'P31' for stable phosphorus, 'P32' for radioactive control
        
    Returns:
        Configured Model6Parameters
    """
    params = Model6Parameters()
    
    # Set isotope fraction
    if isotope == 'P31':
        params.environment.fraction_P31 = 1.0  # 100% P31
    elif isotope == 'P32':
        params.environment.fraction_P31 = 0.0  # 100% P32
    else:
        raise ValueError(f"Unknown isotope: {isotope}")
    
    # Disable EM coupling for cleaner baseline experiment
    params.em_coupling_enabled = False
    
    return params


def run_simulation_steps(model: Model6QuantumSynapse, duration: float, 
                         stimulus: Dict, dt: Optional[float] = None) -> int:
    """
    Run simulation for specified duration with given stimulus.
    
    Args:
        model: The Model6QuantumSynapse instance
        duration: Time to simulate (seconds)
        stimulus: Stimulus dictionary for model.step()
        dt: Timestep (uses model.dt if None)
        
    Returns:
        Number of steps taken
    """
    if dt is None:
        dt = model.dt
    
    # Use larger timestep during rest periods (no activity)
    is_rest = stimulus.get('voltage', -70e-3) < -60e-3
    if is_rest and duration > 1.0:
        dt_effective = 0.1  # 100ms steps during rest
    else:
        dt_effective = dt  # 1ms during activity
    
    n_steps = int(duration / dt_effective)
    for _ in range(n_steps):
        model.step(dt_effective, stimulus)
    
    return n_steps


def get_current_state(model: Model6QuantumSynapse) -> Dict:
    """
    Extract current state from model for diagnostics.
    """
    metrics = model.get_experimental_metrics()
    
    return {
        'time': model.time,
        'calcium_uM': metrics.get('calcium_peak_uM', 0),
        'dimer_nM': metrics.get('dimer_peak_nM_ct', 0),
        'coherence': metrics.get('coherence_dimer_mean', 0),
        'spine_volume': metrics.get('spine_volume_fold', 1.0),
        'AMPAR_count': metrics.get('AMPAR_count', 80),
        'synaptic_strength': metrics.get('synaptic_strength', 1.0),
        # DDSC state
        'ddsc_triggered': model.ddsc.triggered,
        'ddsc_structural_drive': model.ddsc.get_structural_drive(),
        'ddsc_integrated': model.ddsc.integrated_ddsc,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_trial(
    delay: float,
    measurement_time: float,
    isotope: str = 'P31',
    verbose: bool = False
) -> Dict:
    """
    Run single trial of interpulse delay experiment.
    
    Protocol:
    1. Stimulation (200ms) - depolarization creates dimers, establishes eligibility
    2. Delay (variable) - rest, eligibility decays with T2
    3. Plateau (300ms) - checks eligibility, triggers DDSC if above threshold
    4. Wait (measurement_time) - DDSC drives structural changes
    5. Measure - record final plasticity state
    
    Args:
        delay: Time between stimulation and plateau (seconds)
        measurement_time: Time after plateau to measure (seconds)
        isotope: 'P31' or 'P32'
        verbose: Print progress
        
    Returns:
        Dictionary with all results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Trial: {isotope}, delay={delay}s, measure={measurement_time}s")
        print(f"{'='*60}")
    
    # Initialize model with correct isotope
    params = create_model_params(isotope)
    model = Model6QuantumSynapse(params=params)
    dt = model.dt
    
    # Record initial state
    initial_state = get_current_state(model)
    initial_strength = initial_state['synaptic_strength']
    
    if verbose:
        print(f"Initial state: strength={initial_strength:.3f}")
    
    # === PHASE 1: STIMULATION ===
    # Depolarize to open calcium channels, create dimers
    if verbose:
        print(f"\nPhase 1: Stimulation ({STIM_DURATION}s)")
    
    stim_stimulus = {'voltage': STIM_VOLTAGE, 'plateau_potential': False}
    run_simulation_steps(model, STIM_DURATION, stim_stimulus)
    
    post_stim_state = get_current_state(model)
    if verbose:
        print(f"  Post-stim: Ca={post_stim_state['calcium_uM']:.2f}μM, "
              f"dimers={post_stim_state['dimer_nM']:.2f}nM, "
              f"coherence={post_stim_state['coherence']:.3f}")
    
    # === PHASE 2: DELAY ===
    # Return to rest, eligibility decays
    if verbose:
        print(f"\nPhase 2: Delay ({delay}s)")
    
    rest_stimulus = {'voltage': REST_VOLTAGE, 'plateau_potential': False}
    run_simulation_steps(model, delay, rest_stimulus)
    
    pre_plateau_state = get_current_state(model)
    eligibility_at_plateau = pre_plateau_state['coherence']  # Coherence IS eligibility
    
    if verbose:
        print(f"  Pre-plateau: coherence/eligibility={eligibility_at_plateau:.3f}")
    
    # === PHASE 3: PLATEAU ===
    # Check eligibility and trigger DDSC if above threshold
    if verbose:
        print(f"\nPhase 3: Plateau ({PLATEAU_DURATION}s)")
    
    plateau_stimulus = {'voltage': STIM_VOLTAGE, 'plateau_potential': True}
    run_simulation_steps(model, PLATEAU_DURATION, plateau_stimulus)
    
    post_plateau_state = get_current_state(model)
    gate_opened = post_plateau_state['ddsc_triggered']
    
    if verbose:
        print(f"  Gate opened: {gate_opened}")
        if gate_opened:
            print(f"  DDSC eligibility at trigger: {model.ddsc.eligibility_at_trigger:.3f}")
    
    # === PHASE 4: WAIT FOR STRUCTURAL CHANGES ===
    if verbose:
        print(f"\nPhase 4: Consolidation ({measurement_time}s)")
    
    # Return to rest, let DDSC drive structural changes
    rest_stimulus = {'voltage': REST_VOLTAGE, 'plateau_potential': False}
    
    # Run in chunks to track progress
    chunk_size = min(60.0, measurement_time)  # Report every 60s or at end
    elapsed = 0.0
    
    while elapsed < measurement_time:
        chunk = min(chunk_size, measurement_time - elapsed)
        run_simulation_steps(model, chunk, rest_stimulus)
        elapsed += chunk
        
        if verbose and elapsed < measurement_time:
            current = get_current_state(model)
            print(f"  t={elapsed:.0f}s: structural_drive={current['ddsc_structural_drive']:.3f}, "
                  f"spine_vol={current['spine_volume']:.2f}x")
    
    # === PHASE 5: FINAL MEASUREMENT ===
    final_state = get_current_state(model)
    final_strength = final_state['synaptic_strength']
    plasticity_percent = (final_strength - initial_strength) / initial_strength * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULT: Plasticity = {plasticity_percent:+.1f}%")
        print(f"  Initial strength: {initial_strength:.3f}")
        print(f"  Final strength: {final_strength:.3f}")
        print(f"  Spine volume: {final_state['spine_volume']:.2f}x")
        print(f"  AMPAR count: {final_state['AMPAR_count']:.0f}")
        print(f"{'='*60}")
    
    # Collect all results
    return {
        'delay': delay,
        'measurement_time': measurement_time,
        'isotope': isotope,
        'eligibility_at_plateau': eligibility_at_plateau,
        'gate_opened': gate_opened,
        'initial_strength': initial_strength,
        'final_strength': final_strength,
        'plasticity_percent': plasticity_percent,
        'spine_volume': final_state['spine_volume'],
        'AMPAR_count': final_state['AMPAR_count'],
        'ddsc_state': model.ddsc.get_state(),
        'T2_used': 'P31' if model.params.environment.fraction_P31 > 0.5 else 'P32',
    }


def run_experiment(
    delays: List[float],
    measurement_times: List[float],
    isotope: str = 'P31',
    n_trials: int = 1,
    verbose: bool = True
) -> List[Dict]:
    """
    Run full experiment matrix.
    """
    results = []
    total = len(delays) * len(measurement_times) * n_trials
    current = 0
    
    print(f"\nRunning {isotope} experiment: {len(delays)} delays × "
          f"{len(measurement_times)} timepoints × {n_trials} trials = {total} runs")
    
    for delay in delays:
        for mtime in measurement_times:
            for trial in range(n_trials):
                current += 1
                print(f"\n[{current}/{total}] {isotope} delay={delay}s, measure={mtime}s, trial={trial+1}")
                
                result = run_single_trial(
                    delay=delay,
                    measurement_time=mtime,
                    isotope=isotope,
                    verbose=verbose and (trial == 0)  # Verbose for first trial only
                )
                result['trial'] = trial
                results.append(result)
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(p31_results: List[Dict], p32_results: List[Dict], 
                 output_path: str = 'interpulse_delay_results.png'):
    """
    Create publication figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get unique measurement times
    measurement_times = sorted(set(r['measurement_time'] for r in p31_results))
    final_mtime = max(measurement_times)
    
    # Panel A: Plasticity accumulation over time (P31 only)
    ax = axes[0, 0]
    delays_to_plot = sorted(set(r['delay'] for r in p31_results))[:3]  # First 3 delays
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(delays_to_plot)))
    
    for delay, color in zip(delays_to_plot, colors):
        subset = [r for r in p31_results if r['delay'] == delay]
        if len(measurement_times) > 1:
            times = [r['measurement_time'] / 60 for r in subset]
            plasticity = [r['plasticity_percent'] for r in subset]
            ax.plot(times, plasticity, 'o-', color=color, label=f'delay={delay}s', linewidth=2, markersize=8)
        else:
            # Single measurement time - show as bar
            plasticity = np.mean([r['plasticity_percent'] for r in subset])
            ax.bar(delay, plasticity, color=color, width=5, label=f'{plasticity:+.1f}%')
    
    ax.set_xlabel('Measurement Time (minutes)' if len(measurement_times) > 1 else 'Delay (s)', fontsize=12)
    ax.set_ylabel('Plasticity (%)', fontsize=12)
    ax.set_title('A. Plasticity Accumulation (³¹P)', fontsize=14)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel B: Final plasticity vs delay (P31 vs P32)
    ax = axes[0, 1]
    
    # P31 at final measurement time
    p31_final = [r for r in p31_results if r['measurement_time'] == final_mtime]
    p31_delays = [r['delay'] for r in p31_final]
    p31_plasticity = [r['plasticity_percent'] for r in p31_final]
    ax.plot(p31_delays, p31_plasticity, 'o-', color='blue', label='³¹P', linewidth=2, markersize=8)
    
    # P32 at final measurement time
    p32_final = [r for r in p32_results if r['measurement_time'] == final_mtime]
    p32_delays = [r['delay'] for r in p32_final]
    p32_plasticity = [r['plasticity_percent'] for r in p32_final]
    if p32_delays:
        ax.plot(p32_delays, p32_plasticity, 's--', color='red', label='³²P', linewidth=2, markersize=8)
    
    ax.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax.set_ylabel(f'Plasticity at {final_mtime/60:.0f} min (%)', fontsize=12)
    ax.set_title('B. Isotope Comparison', fontsize=14)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel C: Eligibility at plateau vs delay
    ax = axes[1, 0]
    
    p31_elig = [r['eligibility_at_plateau'] for r in p31_final]
    ax.plot(p31_delays, p31_elig, 'o-', color='blue', label='³¹P', linewidth=2, markersize=8)
    
    if p32_final:
        p32_elig = [r['eligibility_at_plateau'] for r in p32_final]
        ax.plot(p32_delays, p32_elig, 's--', color='red', label='³²P', linewidth=2, markersize=8)
    
    ax.axhline(y=0.3, color='green', linestyle=':', label='Threshold', linewidth=2)
    ax.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax.set_ylabel('Eligibility at Plateau', fontsize=12)
    ax.set_title('C. Eligibility Decay', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Panel D: Gate opened vs delay
    ax = axes[1, 1]
    
    p31_gates = [1 if r['gate_opened'] else 0 for r in p31_final]
    ax.bar([d - 2 for d in p31_delays], p31_gates, width=4, color='blue', alpha=0.7, label='³¹P')
    
    if p32_final:
        p32_gates = [1 if r['gate_opened'] else 0 for r in p32_final]
        ax.bar([d + 2 for d in p32_delays], p32_gates, width=4, color='red', alpha=0.7, label='³²P')
    
    ax.set_xlabel('Interpulse Delay (s)', fontsize=12)
    ax.set_ylabel('Gate Opened (0/1)', fontsize=12)
    ax.set_title('D. DDSC Trigger Success', fontsize=14)
    ax.legend()
    ax.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.show()
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("INTERPULSE DELAY EXPERIMENT - PROPER MODEL 6 IMPLEMENTATION")
    print("=" * 70)
    print(f"\nUsing ACTUAL Model 6 physics:")
    print(f"  - Isotope set via params.environment.fraction_P31")
    print(f"  - T2 emerges from quantum_coherence module")
    print(f"  - Eligibility from dimer coherence")
    print(f"  - DDSC gates structural plasticity")
    print(f"\nQuick test mode: {QUICK_TEST}")
    print(f"P31 delays: {P31_DELAYS}")
    print(f"P32 delays: {P32_DELAYS}")
    print(f"Measurement times: {MEASUREMENT_TIMES}")
    
    # Run P31 experiments
    print("\n" + "=" * 70)
    print("RUNNING ³¹P EXPERIMENTS")
    print("=" * 70)
    p31_results = run_experiment(
        delays=P31_DELAYS,
        measurement_times=MEASUREMENT_TIMES,
        isotope='P31',
        n_trials=1,
        verbose=True
    )
    
    # Run P32 experiments  
    print("\n" + "=" * 70)
    print("RUNNING ³²P EXPERIMENTS")
    print("=" * 70)
    p32_results = run_experiment(
        delays=P32_DELAYS,
        measurement_times=MEASUREMENT_TIMES,
        isotope='P32',
        n_trials=1,
        verbose=True
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    final_mtime = max(MEASUREMENT_TIMES)
    
    print(f"\n³¹P Results (at {final_mtime/60:.0f} min):")
    print(f"{'Delay (s)':<12} {'Eligibility':<12} {'Gate':<8} {'Plasticity':<12}")
    print("-" * 50)
    for r in [r for r in p31_results if r['measurement_time'] == final_mtime]:
        print(f"{r['delay']:<12.1f} {r['eligibility_at_plateau']:<12.3f} "
              f"{str(r['gate_opened']):<8} {r['plasticity_percent']:+.1f}%")
    
    print(f"\n³²P Results (at {final_mtime/60:.0f} min):")
    print(f"{'Delay (s)':<12} {'Eligibility':<12} {'Gate':<8} {'Plasticity':<12}")
    print("-" * 50)
    for r in [r for r in p32_results if r['measurement_time'] == final_mtime]:
        print(f"{r['delay']:<12.1f} {r['eligibility_at_plateau']:<12.3f} "
              f"{str(r['gate_opened']):<8} {r['plasticity_percent']:+.1f}%")
    
    # Generate plots
    print("\n>>> Generating figures...")
    output_path = f'interpulse_delay_results_{timestamp}.png'
    plot_results(p31_results, p32_results, output_path)
    
    print("\n>>> Experiment complete!")
    
    return p31_results, p32_results


if __name__ == "__main__":
    main()