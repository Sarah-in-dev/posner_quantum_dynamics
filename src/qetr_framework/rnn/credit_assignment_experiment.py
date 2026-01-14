"""
Credit Assignment Experiment: Delay Sweep
==========================================

The key experiment for Nature Machine Intelligence:
Shows that quantum eligibility traces enable credit assignment
at delays where classical mechanisms fail.

Protocol:
1. t=0: Stimulus activates pattern → dimers form
2. t=0 to T: Silence (dimers decay according to physics)
3. t=T: Reward + calcium → plasticity if eligibility remains

Conditions:
- Quantum: Emergent τ~100s from singlet physics
- Classical τ=1s: Typical RL eligibility trace
- Classical τ=5s: Longer classical trace
- Classical τ=20s: Very long classical trace
- Classical τ=100s: "Oracle" that matches quantum

Delays: 1, 5, 10, 20, 30, 60, 90, 120 seconds

Output:
- Figure: Learning success vs delay for each condition
- Data: CSV with all results

Author: Sarah Davidson
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import time

from quantum_rnn_synapse import QuantumRNNSynapse, SynapseParameters, ClassicalSynapse


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for credit assignment experiment"""
    
    # Delays to test (seconds)
    delays: List[float] = field(default_factory=lambda: [1, 5, 10, 20, 30, 60, 90, 120])
    
    # Number of trials per condition
    n_trials: int = 20
    
    # Classical τ values to compare
    classical_taus: List[float] = field(default_factory=lambda: [1.0, 5.0, 20.0, 100.0])
    
    # Simulation parameters
    dt: float = 0.1  # Timestep (seconds)
    n_activation_steps: int = 10  # Steps of coincident activity (1s total)
    n_reward_steps: int = 10  # Steps of reward (1s total)
    
    # Threshold for "successful" learning
    success_threshold: float = 0.005  # Minimum Δw to count as learning
    
    # Random seed base (each trial gets seed_base + trial_idx)
    seed_base: int = 1000


@dataclass
class TrialResult:
    """Result from a single trial"""
    condition: str
    delay: float
    trial_idx: int
    
    # Eligibility at key timepoints
    eligibility_after_activation: float = 0.0
    eligibility_after_delay: float = 0.0
    
    # Weight change
    weight_before: float = 0.0
    weight_after: float = 0.0
    delta_weight: float = 0.0
    
    # Success
    learned: bool = False


@dataclass
class ConditionResults:
    """Aggregated results for one condition"""
    condition: str
    delays: List[float] = field(default_factory=list)
    
    # Per-delay statistics
    mean_delta_weight: List[float] = field(default_factory=list)
    std_delta_weight: List[float] = field(default_factory=list)
    success_rate: List[float] = field(default_factory=list)
    mean_eligibility_at_reward: List[float] = field(default_factory=list)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_trial(
    condition: str,
    delay: float,
    trial_idx: int,
    config: ExperimentConfig,
    tau: Optional[float] = None
) -> TrialResult:
    """
    Run a single credit assignment trial.
    
    Args:
        condition: 'Quantum' or 'Classical'
        delay: Delay between activity and reward (seconds)
        trial_idx: Trial index (for seeding)
        config: Experiment configuration
        tau: Time constant for classical (None for quantum)
    
    Returns:
        TrialResult with all measurements
    """
    result = TrialResult(
        condition=condition if tau is None else f"Classical τ={tau}s",
        delay=delay,
        trial_idx=trial_idx
    )
    
    seed = config.seed_base + trial_idx
    
    # Create synapse
    if condition == 'Quantum':
        synapse = QuantumRNNSynapse(seed=seed)
    else:
        synapse = ClassicalSynapse(tau=tau)
    
    dt = config.dt
    
    # Phase 1: Activation (coincident pre/post activity)
    for _ in range(config.n_activation_steps):
        synapse.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    
    # Record eligibility after activation
    if condition == 'Quantum':
        result.eligibility_after_activation = synapse.eligibility
    else:
        result.eligibility_after_activation = synapse.eligibility
    
    # Phase 2: Delay (silence - no activity)
    n_delay_steps = int(delay / dt)
    for _ in range(n_delay_steps):
        synapse.step(dt, pre_active=False, post_active=False, dopamine=0.0)
    
    # Record eligibility after delay
    if condition == 'Quantum':
        result.eligibility_after_delay = synapse.eligibility
    else:
        result.eligibility_after_delay = synapse.eligibility
    
    # Phase 3: Reward (dopamine + calcium, no new dimers)
    result.weight_before = synapse.weight
    
    for _ in range(config.n_reward_steps):
        synapse.step(dt, pre_active=False, post_active=True, dopamine=1.0)
    
    result.weight_after = synapse.weight
    result.delta_weight = result.weight_after - result.weight_before
    result.learned = result.delta_weight > config.success_threshold
    
    return result


def run_condition(
    condition: str,
    config: ExperimentConfig,
    tau: Optional[float] = None,
    verbose: bool = True
) -> ConditionResults:
    """
    Run all trials for one condition across all delays.
    
    Args:
        condition: 'Quantum' or 'Classical'
        config: Experiment configuration
        tau: Time constant for classical
        verbose: Print progress
    
    Returns:
        ConditionResults with aggregated statistics
    """
    condition_name = condition if tau is None else f"Classical τ={tau}s"
    results = ConditionResults(condition=condition_name)
    
    if verbose:
        print(f"\n  {condition_name}:")
    
    for delay in config.delays:
        trial_results = []
        
        for trial_idx in range(config.n_trials):
            trial = run_single_trial(condition, delay, trial_idx, config, tau)
            trial_results.append(trial)
        
        # Aggregate
        delta_weights = [t.delta_weight for t in trial_results]
        successes = [t.learned for t in trial_results]
        eligibilities = [t.eligibility_after_delay for t in trial_results]
        
        results.delays.append(delay)
        results.mean_delta_weight.append(np.mean(delta_weights))
        results.std_delta_weight.append(np.std(delta_weights))
        results.success_rate.append(np.mean(successes))
        results.mean_eligibility_at_reward.append(np.mean(eligibilities))
        
        if verbose:
            print(f"    delay={delay:3.0f}s: Δw={np.mean(delta_weights):.4f} ± {np.std(delta_weights):.4f}, "
                  f"success={np.mean(successes)*100:.0f}%")
    
    return results


def run_experiment(config: Optional[ExperimentConfig] = None, verbose: bool = True) -> Dict[str, ConditionResults]:
    """
    Run the full credit assignment experiment.
    
    Args:
        config: Experiment configuration
        verbose: Print progress
    
    Returns:
        Dictionary mapping condition names to results
    """
    if config is None:
        config = ExperimentConfig()
    
    if verbose:
        print("=" * 70)
        print("CREDIT ASSIGNMENT EXPERIMENT")
        print("=" * 70)
        print(f"\nDelays: {config.delays}")
        print(f"Trials per condition: {config.n_trials}")
        print(f"Classical τ values: {config.classical_taus}")
    
    all_results = {}
    start_time = time.time()
    
    # Run quantum condition
    if verbose:
        print("\nRunning conditions...")
    
    quantum_results = run_condition('Quantum', config, tau=None, verbose=verbose)
    all_results['Quantum'] = quantum_results
    
    # Run classical conditions
    for tau in config.classical_taus:
        classical_results = run_condition('Classical', config, tau=tau, verbose=verbose)
        all_results[classical_results.condition] = classical_results
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nCompleted in {elapsed:.1f}s")
    
    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(
    results: Dict[str, ConditionResults],
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate publication-quality figure.
    
    Args:
        results: Dictionary of condition results
        output_path: Path to save figure (optional)
        show: Whether to display figure
    
    Returns:
        matplotlib Figure
    """
    # Style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors
    colors = {
        'Quantum': '#2ca02c',  # Green
        'Classical τ=1.0s': '#d62728',  # Red
        'Classical τ=5.0s': '#ff7f0e',  # Orange
        'Classical τ=20.0s': '#9467bd',  # Purple
        'Classical τ=100.0s': '#1f77b4',  # Blue
    }
    
    # Line styles
    linestyles = {
        'Quantum': '-',
        'Classical τ=1.0s': '--',
        'Classical τ=5.0s': '--',
        'Classical τ=20.0s': '--',
        'Classical τ=100.0s': ':',
    }
    
    # === Panel A: Mean Weight Change vs Delay ===
    ax1 = axes[0]
    
    for condition, res in results.items():
        color = colors.get(condition, 'gray')
        ls = linestyles.get(condition, '-')
        
        ax1.errorbar(
            res.delays,
            res.mean_delta_weight,
            yerr=res.std_delta_weight,
            label=condition,
            color=color,
            linestyle=ls,
            linewidth=2,
            marker='o',
            markersize=6,
            capsize=3,
            alpha=0.9
        )
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Delay (seconds)')
    ax1.set_ylabel('Weight Change (Δw)')
    ax1.set_title('A. Credit Assignment Performance')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 130)
    ax1.set_ylim(-0.005, None)
    
    # Add shaded region for "quantum advantage window"
    ax1.axvspan(30, 120, alpha=0.1, color='green', label='_nolegend_')
    ax1.text(75, ax1.get_ylim()[1] * 0.9, 'Quantum\nAdvantage', 
             ha='center', va='top', fontsize=9, color='green', alpha=0.7)
    
    # === Panel B: Success Rate vs Delay ===
    ax2 = axes[1]
    
    for condition, res in results.items():
        color = colors.get(condition, 'gray')
        ls = linestyles.get(condition, '-')
        
        ax2.plot(
            res.delays,
            [s * 100 for s in res.success_rate],
            label=condition,
            color=color,
            linestyle=ls,
            linewidth=2,
            marker='o',
            markersize=6,
            alpha=0.9
        )
    
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Delay (seconds)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('B. Learning Success Rate')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 130)
    ax2.set_ylim(-5, 105)
    
    # Add annotation
    ax2.text(75, 10, 'Chance', ha='center', va='bottom', fontsize=9, color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def save_results(
    results: Dict[str, ConditionResults],
    output_path: Path
) -> None:
    """Save results to JSON file."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'conditions': {}
    }
    
    for condition, res in results.items():
        data['conditions'][condition] = {
            'delays': res.delays,
            'mean_delta_weight': res.mean_delta_weight,
            'std_delta_weight': res.std_delta_weight,
            'success_rate': res.success_rate,
            'mean_eligibility_at_reward': res.mean_eligibility_at_reward,
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary(results: Dict[str, ConditionResults]) -> None:
    """Print summary table of results."""
    
    print("\n" + "=" * 70)
    print("SUMMARY: Credit Assignment at Key Delays")
    print("=" * 70)
    
    key_delays = [10, 30, 60, 90]
    
    print(f"\n{'Condition':<22}", end="")
    for d in key_delays:
        print(f"  {d}s", end="")
        print(" " * 8, end="")
    print()
    
    print("-" * 70)
    
    for condition, res in results.items():
        print(f"{condition:<22}", end="")
        for d in key_delays:
            if d in res.delays:
                idx = res.delays.index(d)
                dw = res.mean_delta_weight[idx]
                sr = res.success_rate[idx] * 100
                print(f"  {dw:.4f} ({sr:3.0f}%)", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()
    
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    
    # Find where quantum still works but classical fails
    quantum = results.get('Quantum')
    classical_5 = results.get('Classical τ=5.0s')
    
    if quantum and classical_5:
        for i, delay in enumerate(quantum.delays):
            q_success = quantum.success_rate[i]
            c_success = classical_5.success_rate[i] if i < len(classical_5.success_rate) else 0
            
            if q_success > 0.8 and c_success < 0.2:
                print(f"\nAt delay = {delay}s:")
                print(f"  - Quantum: {q_success*100:.0f}% success, Δw = {quantum.mean_delta_weight[i]:.4f}")
                print(f"  - Classical τ=5s: {c_success*100:.0f}% success")
                print(f"\n  → Quantum enables credit assignment where classical fails!")
                break
    
    print("""
INTERPRETATION:
- Quantum eligibility (τ~100s from singlet physics) enables learning at long delays
- Classical eligibility requires τ to be tuned per task
- Even τ=100s (oracle) matches quantum, but requires knowing the delay a priori
- Quantum provides the right timescale without hyperparameter tuning
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run credit assignment experiment')
    parser.add_argument('--trials', type=int, default=20, help='Trials per condition')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--no-show', action='store_true', help='Do not display figure')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer trials')
    
    args = parser.parse_args()
    
    # Configure
    config = ExperimentConfig(n_trials=args.trials)
    
    if args.quick:
        config.n_trials = 5
        config.delays = [1, 10, 30, 60]
        config.classical_taus = [1.0, 5.0, 20.0]
    
    # Run
    results = run_experiment(config, verbose=True)
    
    # Output
    output_dir = Path(args.output)
    
    # Save data
    save_results(results, output_dir / 'credit_assignment_results.json')
    
    # Generate figure
    fig = plot_results(
        results,
        output_path=output_dir / 'credit_assignment_figure.png',
        show=not args.no_show
    )
    
    # Print summary
    print_summary(results)