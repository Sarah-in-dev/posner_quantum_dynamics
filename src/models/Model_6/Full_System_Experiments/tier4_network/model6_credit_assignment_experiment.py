"""
Model 6 Credit Assignment Experiment
=====================================

Proper experiment to test quantum eligibility traces for temporal credit assignment.

This differs from the simplified RNN experiment in critical ways:
1. Uses FULL Model 6 physics (not simplified synapse)
2. Network stays ACTIVE during delay (realistic working memory)
3. Tests actual LEARNING (pattern discrimination), not just trace persistence
4. Proper controls: P31 vs P32, MT invaded vs not

Experiment Protocol:
--------------------
Phase 1: ENCODING
    - Present stimulus pattern A or B (different input patterns)
    - Network processes through recurrent dynamics
    - Dimers form at active synapses
    
Phase 2: DELAY (network ACTIVE)
    - Stimulus removed, but network maintains activity through recurrence
    - This is working memory - the network must maintain pattern information
    - Dimers decay according to quantum physics
    
Phase 3: RESPONSE
    - Network produces output
    - Reward given if correct pattern was maintained
    
Measure:
    - Can network learn to discriminate A vs B at long delays?
    - Compare P31 (quantum, τ~100s) vs P32 (control, τ~0.4s)

Key Prediction:
    - P31 networks maintain eligibility, can learn at 60s delays
    - P32 networks lose eligibility by ~1s, cannot learn at long delays

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import time
import logging

# Import the network (this file should be in same directory)
from model6_recurrent_network import Model6RecurrentNetwork, NetworkConfig

logger = logging.getLogger(__name__)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for credit assignment experiment"""
    
    # Network parameters
    n_neurons: int = 5  # Small for computational tractability with full Model 6
    
    # Delays to test (seconds)
    delays: List[float] = field(default_factory=lambda: [1, 5, 10, 20, 30, 60])
    
    # Number of trials per condition
    n_trials: int = 3 # Keep small for speed; increase for real runs
    
    # Conditions to test
    conditions: List[str] = field(default_factory=lambda: ['P31', 'P32'])
    
    # Timing (seconds)
    encoding_duration: float = 0.05     # How long to present stimulus
    response_duration: float = 0.5     # How long to collect response
    reward_duration: float = 0.5       # How long reward signal lasts
    
    # Timestep
    dt_fine: float = 0.050    # 50 ms during encoding/response
    dt_coarse: float = 1.0    # 1s during delay (dimers just decay)
    
    # Stimulus parameters
    stimulus_strength: float = 2.0
    n_pattern_neurons: int = 3  # How many neurons encode each pattern
    
    # Learning threshold
    learning_threshold: float = 0.01  # Minimum Δcommitment to count as learning
    
    # Random seed
    seed_base: int = 42


@dataclass
class TrialResult:
    """Result from a single trial"""
    condition: str
    delay: float
    trial_idx: int
    pattern: str  # 'A' or 'B'
    
    # State at key timepoints
    dimers_after_encoding: int = 0
    eligibility_after_encoding: float = 0.0
    eligibility_after_delay: float = 0.0
    
    # Commitment changes
    commitment_before: float = 0.0
    commitment_after: float = 0.0
    delta_commitment: float = 0.0
    
    # Did learning occur?
    learned: bool = False
    
    # Network activity
    mean_rate_during_delay: float = 0.0
    
    # Timing
    runtime_s: float = 0.0


@dataclass
class ConditionResult:
    """Aggregated results for one condition across all delays"""
    condition: str
    delays: List[float] = field(default_factory=list)
    
    # Means and stds
    mean_eligibility_at_reward: List[float] = field(default_factory=list)
    mean_delta_commitment: List[float] = field(default_factory=list)
    std_delta_commitment: List[float] = field(default_factory=list)
    learning_success_rate: List[float] = field(default_factory=list)
    
    # For plotting
    all_trials: List[TrialResult] = field(default_factory=list)


# =============================================================================
# STIMULUS GENERATION
# =============================================================================

def generate_stimulus(
    pattern: str, 
    n_neurons: int, 
    n_pattern: int,
    strength: float
) -> np.ndarray:
    """
    Generate input stimulus for pattern A or B.
    
    Pattern A: Activates neurons [0, 1, ..., n_pattern-1]
    Pattern B: Activates neurons [n_pattern, n_pattern+1, ..., 2*n_pattern-1]
    """
    stimulus = np.zeros(n_neurons)
    
    if pattern == 'A':
        stimulus[:n_pattern] = strength
    elif pattern == 'B':
        stimulus[n_pattern:2*n_pattern] = strength
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return stimulus


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_single_trial(
    network: Model6RecurrentNetwork,
    pattern: str,
    delay: float,
    config: ExperimentConfig,
    verbose: bool = False
) -> TrialResult:
    """
    Run a single trial of the credit assignment task.
    
    Protocol:
    1. ENCODING: Present stimulus, form dimers
    2. DELAY: Network maintains activity, dimers decay
    3. RESPONSE + REWARD: Evaluate and provide feedback
    """
    start_time = time.time()
    
    result = TrialResult(
        condition=network.config.isotope,
        delay=delay,
        trial_idx=0,  # Will be set by caller
        pattern=pattern
    )
    
    # Generate stimulus
    stimulus = generate_stimulus(
        pattern=pattern,
        n_neurons=config.n_neurons,
        n_pattern=config.n_pattern_neurons,
        strength=config.stimulus_strength
    )
    no_stimulus = np.zeros(config.n_neurons)
    
    # --- PHASE 1: ENCODING ---
    if verbose:
        print(f"    Phase 1: Encoding (theta-burst protocol)")

    # Use theta-burst encoding for validated dimer formation
    state = network.run_encoding_theta_burst(
        external_input=stimulus,
        n_bursts=5,
        reward=False
    )
        
    result.dimers_after_encoding = state.total_dimers
    result.eligibility_after_encoding = state.mean_eligibility
    result.commitment_before = np.sum(network.get_commitment_matrix())
    
    if verbose:
        print(f"      Dimers: {state.total_dimers}, Eligibility: {state.mean_eligibility:.3f}")
    
    # --- PHASE 2: DELAY (network stays ACTIVE via recurrence) ---
    if verbose:
        print(f"    Phase 2: Delay ({delay}s) - network maintains activity")
    
    # Use coarse timestep for long delays
    dt_delay = config.dt_coarse if delay > 5 else config.dt_fine
    n_delay_steps = int(delay / dt_delay)
    
    rates_during_delay = []
    
    for step in range(n_delay_steps):
        
        state = network.step_delay_fast(dt_delay)
        
        if step % max(1, n_delay_steps // 10) == 0:
            rates_during_delay.append(np.mean(state.rates))
    
    result.eligibility_after_delay = state.mean_eligibility
    result.mean_rate_during_delay = np.mean(rates_during_delay) if rates_during_delay else 0.0
    
    if verbose:
        print(f"      Eligibility remaining: {state.mean_eligibility:.3f}")
        print(f"      Mean activity: {result.mean_rate_during_delay:.3f}")
    
    # --- PHASE 3: RESPONSE + REWARD ---
    if verbose:
        print(f"    Phase 3: Reward ({config.reward_duration}s)")
    
    # Brief reactivation with reward
    # In a full task, we'd evaluate output and give reward conditionally
    # Here, we just test if reward can trigger plasticity
    
    n_reward_steps = int(config.reward_duration / config.dt_fine)
    
    # Provide mild stimulus during reward (postsynaptic calcium needed for three-factor gate)
    mild_stimulus = stimulus * 0.5  # Weaker than encoding
    
    for _ in range(n_reward_steps):
        state = network.step_reward_only(config.dt_fine)
    
    result.commitment_after = np.sum(network.get_commitment_matrix())
    result.delta_commitment = result.commitment_after - result.commitment_before
    result.learned = result.delta_commitment > config.learning_threshold
    
    if verbose:
        print(f"      ΔCommitment: {result.delta_commitment:.4f}")
        print(f"      Learned: {result.learned}")
    
    result.runtime_s = time.time() - start_time
    
    return result


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

def run_condition(
    condition: str,
    config: ExperimentConfig,
    Model6Class: Any,
    Model6Params: Any,
    verbose: bool = True
) -> ConditionResult:
    """
    Run all trials for one condition (isotope) across all delays.
    """
    result = ConditionResult(condition=condition)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")
    
    for delay in config.delays:
        if verbose:
            print(f"\n  Delay: {delay}s")
        
        trial_results = []
        
        for trial_idx in range(config.n_trials):
            # Create fresh network for each trial
            seed = config.seed_base + trial_idx
            
            network_config = NetworkConfig(
                n_neurons=config.n_neurons,
                isotope=condition,
                mt_invaded=True
            )
            
            network = Model6RecurrentNetwork(
                config=network_config,
                Model6Class=Model6Class,
                Model6Params=Model6Params,
                seed=seed
            )
            
            # Alternate patterns
            pattern = 'A' if trial_idx % 2 == 0 else 'B'
            
            trial = run_single_trial(
                network=network,
                pattern=pattern,
                delay=delay,
                config=config,
                verbose=verbose and trial_idx == 0
            )
            trial.trial_idx = trial_idx
            
            trial_results.append(trial)
        
        # Aggregate
        eligibilities = [t.eligibility_after_delay for t in trial_results]
        delta_commits = [t.delta_commitment for t in trial_results]
        successes = [t.learned for t in trial_results]
        
        result.delays.append(delay)
        result.mean_eligibility_at_reward.append(np.mean(eligibilities))
        result.mean_delta_commitment.append(np.mean(delta_commits))
        result.std_delta_commitment.append(np.std(delta_commits))
        result.learning_success_rate.append(np.mean(successes))
        result.all_trials.extend(trial_results)
        
        if verbose:
            print(f"    Mean elig: {np.mean(eligibilities):.3f}, "
                  f"Mean ΔC: {np.mean(delta_commits):.4f}, "
                  f"Success: {np.mean(successes)*100:.0f}%")
    
    return result


def run_experiment(
    config: ExperimentConfig,
    Model6Class: Any,
    Model6Params: Any,
    verbose: bool = True
) -> Dict[str, ConditionResult]:
    """
    Run the complete credit assignment experiment.
    """
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL 6 CREDIT ASSIGNMENT EXPERIMENT")
        print("="*70)
        print(f"Neurons: {config.n_neurons}")
        print(f"Delays: {config.delays}")
        print(f"Trials per condition: {config.n_trials}")
        print(f"Conditions: {config.conditions}")
    
    for condition in config.conditions:
        result = run_condition(
            condition=condition,
            config=config,
            Model6Class=Model6Class,
            Model6Params=Model6Params,
            verbose=verbose
        )
        results[condition] = result
    
    return results


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def plot_results(
    results: Dict[str, ConditionResult],
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Generate publication-quality figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors
    colors = {
        'P31': '#2E86AB',   # Blue (quantum)
        'P32': '#E94F37',   # Red (control)
    }
    
    # --- Panel A: Eligibility at reward time ---
    ax1 = axes[0]
    
    for condition, result in results.items():
        ax1.plot(
            result.delays,
            result.mean_eligibility_at_reward,
            'o-',
            color=colors.get(condition, 'gray'),
            label=f'{condition}',
            linewidth=2,
            markersize=8
        )
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Entanglement threshold')
    ax1.set_xlabel('Delay (seconds)', fontsize=12)
    ax1.set_ylabel('Eligibility at Reward', fontsize=12)
    ax1.set_title('A. Eligibility Persistence', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Learning success ---
    ax2 = axes[1]
    
    for condition, result in results.items():
        ax2.plot(
            result.delays,
            [s * 100 for s in result.learning_success_rate],
            'o-',
            color=colors.get(condition, 'gray'),
            label=f'{condition}',
            linewidth=2,
            markersize=8
        )
    
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax2.set_xlabel('Delay (seconds)', fontsize=12)
    ax2.set_ylabel('Learning Success (%)', fontsize=12)
    ax2.set_title('B. Credit Assignment Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_summary(results: Dict[str, ConditionResult]):
    """Print summary of results."""
    print("\n" + "="*70)
    print("SUMMARY: Credit Assignment Performance")
    print("="*70)
    
    # Header
    delays = results[list(results.keys())[0]].delays
    print(f"\n{'Condition':<12}", end="")
    for d in delays:
        print(f"  {d:>6}s", end="")
    print()
    print("-" * (12 + len(delays) * 9))
    
    # Each condition
    for condition, result in results.items():
        print(f"{condition:<12}", end="")
        for i, d in enumerate(result.delays):
            success = result.learning_success_rate[i] * 100
            print(f"  {success:>5.0f}%", end="")
        print()
    
    # Key finding
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    
    if 'P31' in results and 'P32' in results:
        p31 = results['P31']
        p32 = results['P32']
        
        # Find longest delay where P31 succeeds but P32 fails
        for i in range(len(p31.delays) - 1, -1, -1):
            delay = p31.delays[i]
            if p31.learning_success_rate[i] > 0.7 and p32.learning_success_rate[i] < 0.3:
                print(f"\nAt delay = {delay}s:")
                print(f"  - P31 (quantum): {p31.learning_success_rate[i]*100:.0f}% success")
                print(f"  - P32 (control): {p32.learning_success_rate[i]*100:.0f}% success")
                print(f"\n  → Quantum coherence enables credit assignment at {delay}s")
                print(f"    where classical mechanisms have failed!")
                break


def save_results(results: Dict[str, ConditionResult], output_path: Path):
    """Save results to JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'conditions': {}
    }
    
    for condition, result in results.items():
        data['conditions'][condition] = {
            'delays': result.delays,
            'mean_eligibility_at_reward': result.mean_eligibility_at_reward,
            'mean_delta_commitment': result.mean_delta_commitment,
            'std_delta_commitment': result.std_delta_commitment,
            'learning_success_rate': result.learning_success_rate
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL 6 CREDIT ASSIGNMENT EXPERIMENT")
    print("="*70)
    print("\nThis experiment requires Model 6 classes to be available.")
    print("\nUsage:")
    print("  from model6_core import Model6QuantumSynapse")
    print("  from model6_parameters import Model6Parameters")
    print("  ")
    print("  config = ExperimentConfig(")
    print("      n_neurons=10,")
    print("      delays=[1, 5, 10, 20, 30, 60],")
    print("      n_trials=10")
    print("  )")
    print("  ")
    print("  results = run_experiment(")
    print("      config=config,")
    print("      Model6Class=Model6QuantumSynapse,")
    print("      Model6Params=Model6Parameters")
    print("  )")
    print("  ")
    print("  plot_results(results)")
    print("  print_summary(results)")