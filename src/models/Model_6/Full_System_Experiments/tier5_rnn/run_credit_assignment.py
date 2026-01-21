"""
Credit Assignment Experiment for Quantum RNN
=============================================

The key experiment demonstrating that quantum eligibility traces
enable temporal credit assignment where classical mechanisms fail.

Protocol:
---------
1. ENCODING (t=0 to t=1s):
   - External stimulus activates subset of neurons
   - Coincident activity → dimers form at active synapses
   - Eligibility created

2. DELAY (t=1s to t=1+T):
   - Stimulus removed
   - Network activity may persist via recurrence
   - Dimers persist (P31) or decay (P32)
   
3. REWARD (t=1+T):
   - Dopamine signal arrives
   - Three-factor gate evaluated
   - Synapses with eligibility + calcium → commit

Predictions:
-----------
- P31 (τ~100s): Learn at delays up to ~60s
- P32 (τ~0.4s): Fail at delays > ~1s
- Classical τ=5s: Fail at delays > ~10s
- Classical τ=100s: "Oracle" - matches P31 but requires tuning

This demonstrates that quantum physics provides the ~100s window
without requiring hyperparameter tuning.

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for credit assignment experiment"""
    
    # Network size
    n_neurons: int = 10
    n_pattern_neurons: int = 3  # How many neurons encode the pattern
    
    # Delays to test (seconds)
    delays: List[float] = field(default_factory=lambda: [1, 5, 10, 20, 30, 60])
    
    # Conditions
    conditions: List[str] = field(default_factory=lambda: ['P31', 'P32'])
    
    # Classical baselines (τ values)
    classical_taus: List[float] = field(default_factory=lambda: [5.0, 100.0])
    
    # Trials per condition
    n_trials: int = 10
    
    # Timing
    encoding_duration: float = 1.0   # seconds
    reward_duration: float = 0.5     # seconds
    
    # Timesteps
    dt: float = 0.01                 # 10 ms during encoding/reward
    dt_delay: float = 0.1            # 100 ms during delay (faster)
    
    # Stimulus strength
    stimulus_strength: float = 2.0
    
    # Learning threshold
    learning_threshold: float = 0.01  # Min Δcommitment to count as learned
    
    # Random seed
    seed_base: int = 42


@dataclass
class TrialResult:
    """Result from a single trial"""
    condition: str
    delay: float
    trial_idx: int
    
    # State at key timepoints
    dimers_after_encoding: int = 0
    eligibility_after_encoding: float = 0.0
    eligibility_after_delay: float = 0.0
    
    # Learning outcome
    commitment_before: float = 0.0
    commitment_after: float = 0.0
    delta_commitment: float = 0.0
    learned: bool = False
    
    # Timing
    runtime_s: float = 0.0


@dataclass
class ConditionResult:
    """Aggregated results for one condition"""
    condition: str
    delays: List[float] = field(default_factory=list)
    
    # Statistics per delay
    mean_eligibility_at_reward: List[float] = field(default_factory=list)
    mean_delta_commitment: List[float] = field(default_factory=list)
    std_delta_commitment: List[float] = field(default_factory=list)
    success_rate: List[float] = field(default_factory=list)
    
    # All trials
    all_trials: List[TrialResult] = field(default_factory=list)


# =============================================================================
# TRIAL EXECUTION
# =============================================================================

def run_quantum_trial(
    rnn,  # QuantumRNN instance
    delay: float,
    config: ExperimentConfig,
    trial_idx: int,
    verbose: bool = False
) -> TrialResult:
    """
    Run a single trial with quantum RNN.
    """
    start_time = time.time()
    
    result = TrialResult(
        condition=rnn.config.isotope,
        delay=delay,
        trial_idx=trial_idx
    )
    
    # Reset network
    rnn.reset_synapses()
    
    # Generate stimulus (activates first n_pattern neurons)
    stimulus = np.zeros(config.n_neurons)
    stimulus[:config.n_pattern_neurons] = config.stimulus_strength
    
    # --- PHASE 1: ENCODING ---
    if verbose:
        print(f"    Phase 1: Encoding ({config.encoding_duration}s)")
    
    n_encoding_steps = int(config.encoding_duration / config.dt)
    for step in range(n_encoding_steps):
        state = rnn.step(config.dt, stimulus, reward=False)
    
    result.dimers_after_encoding = state.total_dimers
    result.eligibility_after_encoding = state.mean_eligibility
    result.commitment_before = np.sum(rnn.get_commitment_matrix())
    
    if verbose:
        print(f"      Dimers: {state.total_dimers}, Eligibility: {state.mean_eligibility:.3f}")
    
    # --- PHASE 2: DELAY ---
    if verbose:
        print(f"    Phase 2: Delay ({delay}s)")
    
    no_stimulus = np.zeros(config.n_neurons)
    n_delay_steps = int(delay / config.dt_delay)
    
    for step in range(n_delay_steps):
        state = rnn.step(config.dt_delay, no_stimulus, reward=False)
    
    result.eligibility_after_delay = state.mean_eligibility
    
    if verbose:
        print(f"      Eligibility remaining: {state.mean_eligibility:.3f}")
    
    # --- PHASE 3: REWARD ---
    if verbose:
        print(f"    Phase 3: Reward ({config.reward_duration}s)")
    
    # Mild reactivation during reward (needed for calcium/three-factor gate)
    mild_stimulus = stimulus * 0.3
    n_reward_steps = int(config.reward_duration / config.dt)
    
    for step in range(n_reward_steps):
        state = rnn.step(config.dt, mild_stimulus, reward=True)
    
    result.commitment_after = np.sum(rnn.get_commitment_matrix())
    result.delta_commitment = result.commitment_after - result.commitment_before
    result.learned = result.delta_commitment > config.learning_threshold
    
    if verbose:
        print(f"      ΔCommitment: {result.delta_commitment:.4f}, Learned: {result.learned}")
    
    result.runtime_s = time.time() - start_time
    
    return result


def run_classical_trial(
    rnn,  # ClassicalRNN instance
    delay: float,
    config: ExperimentConfig,
    trial_idx: int,
    verbose: bool = False
) -> TrialResult:
    """
    Run a single trial with classical RNN.
    """
    start_time = time.time()
    
    result = TrialResult(
        condition=f"Classical τ={rnn.tau}s",
        delay=delay,
        trial_idx=trial_idx
    )
    
    # Reset
    rnn.reset()
    rnn.committed = np.zeros_like(rnn.committed)
    
    # Stimulus
    stimulus = np.zeros(config.n_neurons)
    stimulus[:config.n_pattern_neurons] = config.stimulus_strength
    
    # Encoding
    n_encoding_steps = int(config.encoding_duration / config.dt)
    for _ in range(n_encoding_steps):
        rnn.step(config.dt, stimulus, reward=False)
    
    result.eligibility_after_encoding = np.mean(rnn.eligibility)
    result.commitment_before = np.sum(rnn.committed)
    
    # Delay
    no_stimulus = np.zeros(config.n_neurons)
    n_delay_steps = int(delay / config.dt_delay)
    for _ in range(n_delay_steps):
        rnn.step(config.dt_delay, no_stimulus, reward=False)
    
    result.eligibility_after_delay = np.mean(rnn.eligibility)
    
    # Reward
    mild_stimulus = stimulus * 0.3
    n_reward_steps = int(config.reward_duration / config.dt)
    for _ in range(n_reward_steps):
        rnn.step(config.dt, mild_stimulus, reward=True)
    
    result.commitment_after = np.sum(rnn.committed)
    result.delta_commitment = result.commitment_after - result.commitment_before
    result.learned = result.delta_commitment > 0
    
    result.runtime_s = time.time() - start_time
    
    return result


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class CreditAssignmentExperiment:
    """
    Runs the complete credit assignment experiment.
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 SynapseClass: Any,
                 SynapseParams: Any,
                 verbose: bool = True):
        self.config = config
        self.SynapseClass = SynapseClass
        self.SynapseParams = SynapseParams
        self.verbose = verbose
        
        # Import RNN classes
        from rnn.quantum_rnn import QuantumRNN, ClassicalRNN, RNNConfig
        self.QuantumRNN = QuantumRNN
        self.ClassicalRNN = ClassicalRNN
        self.RNNConfig = RNNConfig
    
    def _create_quantum_rnn(self, isotope: str, seed: int):
        """Create quantum RNN with specified isotope"""
        config = self.RNNConfig(
            n_neurons=self.config.n_neurons,
            isotope=isotope
        )
        return self.QuantumRNN(
            config=config,
            SynapseClass=self.SynapseClass,
            SynapseParams=self.SynapseParams,
            seed=seed
        )
    
    def _create_classical_rnn(self, tau: float, seed: int):
        """Create classical RNN with specified τ"""
        return self.ClassicalRNN(
            n_neurons=self.config.n_neurons,
            tau=tau,
            seed=seed
        )
    
    def run_condition(self, condition: str, tau: Optional[float] = None) -> ConditionResult:
        """Run all trials for one condition"""
        
        condition_name = condition if tau is None else f"Classical τ={tau}s"
        result = ConditionResult(condition=condition_name)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Condition: {condition_name}")
            print(f"{'='*60}")
        
        for delay in self.config.delays:
            if self.verbose:
                print(f"\n  Delay: {delay}s")
            
            trial_results = []
            
            for trial_idx in range(self.config.n_trials):
                seed = self.config.seed_base + trial_idx
                
                if tau is None:
                    # Quantum RNN
                    rnn = self._create_quantum_rnn(condition, seed)
                    trial = run_quantum_trial(
                        rnn, delay, self.config, trial_idx,
                        verbose=self.verbose and trial_idx == 0
                    )
                else:
                    # Classical RNN
                    rnn = self._create_classical_rnn(tau, seed)
                    trial = run_classical_trial(
                        rnn, delay, self.config, trial_idx,
                        verbose=self.verbose and trial_idx == 0
                    )
                
                trial_results.append(trial)
            
            # Aggregate statistics
            eligibilities = [t.eligibility_after_delay for t in trial_results]
            delta_commits = [t.delta_commitment for t in trial_results]
            successes = [t.learned for t in trial_results]
            
            result.delays.append(delay)
            result.mean_eligibility_at_reward.append(np.mean(eligibilities))
            result.mean_delta_commitment.append(np.mean(delta_commits))
            result.std_delta_commitment.append(np.std(delta_commits))
            result.success_rate.append(np.mean(successes))
            result.all_trials.extend(trial_results)
            
            if self.verbose:
                print(f"    Mean eligibility: {np.mean(eligibilities):.3f}")
                print(f"    Mean ΔCommitment: {np.mean(delta_commits):.4f}")
                print(f"    Success rate: {np.mean(successes)*100:.0f}%")
        
        return result
    
    def run(self) -> Dict[str, ConditionResult]:
        """Run the complete experiment"""
        
        results = {}
        
        if self.verbose:
            print("\n" + "="*70)
            print("QUANTUM RNN CREDIT ASSIGNMENT EXPERIMENT")
            print("="*70)
            print(f"Neurons: {self.config.n_neurons}")
            print(f"Delays: {self.config.delays}")
            print(f"Trials per condition: {self.config.n_trials}")
        
        # Quantum conditions
        for isotope in self.config.conditions:
            result = self.run_condition(isotope)
            results[isotope] = result
        
        # Classical baselines
        for tau in self.config.classical_taus:
            result = self.run_condition("Classical", tau=tau)
            results[f"Classical τ={tau}s"] = result
        
        return results
    
    def print_summary(self, results: Dict[str, ConditionResult]):
        """Print formatted summary"""
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Header
        delays = self.config.delays
        print(f"\n{'Condition':<20}", end="")
        for d in delays:
            print(f"  {d:>5}s", end="")
        print()
        print("-" * (20 + len(delays) * 8))
        
        # Success rates
        for condition, result in results.items():
            print(f"{condition:<20}", end="")
            for rate in result.success_rate:
                print(f"  {rate*100:>4.0f}%", end="")
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
                p31_success = p31.success_rate[i]
                p32_success = p32.success_rate[i]
                
                if p31_success > 0.7 and p32_success < 0.3:
                    print(f"\nAt delay = {delay}s:")
                    print(f"  - P31 (quantum):  {p31_success*100:.0f}% success")
                    print(f"  - P32 (control):  {p32_success*100:.0f}% success")
                    print()
                    print(f"  → Quantum coherence enables credit assignment at {delay}s")
                    print(f"    where classical mechanisms fail!")
                    break
    
    def plot_results(self, results: Dict[str, ConditionResult], 
                     output_path: Optional[Path] = None):
        """Generate publication figure"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = {
            'P31': '#2E86AB',
            'P32': '#E94F37',
            'Classical τ=5.0s': '#8B8B8B',
            'Classical τ=100.0s': '#FFB703',
        }
        
        # Panel A: Eligibility persistence
        ax = axes[0]
        for condition, result in results.items():
            color = colors.get(condition, 'black')
            ax.plot(result.delays, result.mean_eligibility_at_reward,
                   'o-', color=color, label=condition, linewidth=2, markersize=8)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Delay (seconds)', fontsize=12)
        ax.set_ylabel('Eligibility at Reward', fontsize=12)
        ax.set_title('A. Eligibility Persistence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Panel B: Learning success
        ax = axes[1]
        for condition, result in results.items():
            color = colors.get(condition, 'black')
            success_pct = [s * 100 for s in result.success_rate]
            ax.plot(result.delays, success_pct,
                   'o-', color=color, label=condition, linewidth=2, markersize=8)
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Delay (seconds)', fontsize=12)
        ax.set_ylabel('Learning Success (%)', fontsize=12)
        ax.set_title('B. Credit Assignment Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, results: Dict[str, ConditionResult], path: Path):
        """Save results to JSON"""
        data = {
            'config': {
                'n_neurons': self.config.n_neurons,
                'delays': self.config.delays,
                'n_trials': self.config.n_trials,
            },
            'conditions': {}
        }
        
        for condition, result in results.items():
            data['conditions'][condition] = {
                'delays': result.delays,
                'mean_eligibility_at_reward': result.mean_eligibility_at_reward,
                'mean_delta_commitment': result.mean_delta_commitment,
                'success_rate': result.success_rate
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run credit assignment experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_credit_assignment.py --quick      # Quick validation
    python run_credit_assignment.py --full       # Full run
    python run_credit_assignment.py --smoke      # Ultra-fast smoke test
        """
    )
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true',
                      help='Ultra-fast smoke test (1 trial, 1 delay, 3 neurons)')
    mode.add_argument('--quick', action='store_true',
                      help='Quick validation (2 trials, short delays)')
    mode.add_argument('--full', action='store_true',
                      help='Full run (10 trials, all delays)')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Import Model6 classes
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    
    # Configure based on mode
    if args.smoke:
        config = ExperimentConfig(
            n_neurons=3,
            delays=[1],
            n_trials=1,
            dt=0.05,
            dt_delay=0.5,
            encoding_duration=0.5,
            reward_duration=0.2,
            conditions=['P31'],
            classical_taus=[],
        )
    elif args.quick:
        config = ExperimentConfig(
            n_neurons=5,
            delays=[1, 5, 10],
            n_trials=2,
            dt=0.02,
            dt_delay=0.1,
        )
    else:  # full
        config = ExperimentConfig(
            n_neurons=10,
            delays=[1, 5, 10, 20, 30, 60],
            n_trials=10,
        )
    
    # Run
    exp = CreditAssignmentExperiment(
        config=config,
        SynapseClass=Model6QuantumSynapse,
        SynapseParams=Model6Parameters
    )
    
    results = exp.run()
    exp.print_summary(results)
    exp.plot_results(results)