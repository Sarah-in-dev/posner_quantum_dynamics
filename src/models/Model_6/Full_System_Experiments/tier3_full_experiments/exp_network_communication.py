#!/usr/bin/env python3
"""
Experiment: Network Communication via Entanglement (Gap 4)
==========================================================

Tests whether quantum entanglement provides a coordination advantage
using the ACTUAL Model 6 physics - MultiSynapseNetwork with 
use_correlated_sampling flag.

Three Conditions:
1. ENTANGLED: MultiSynapseNetwork with use_correlated_sampling=True
   - Full quantum physics (dimers, T₂ ~100s for P31)
   - Correlated eligibility sampling at reward time
   - PREDICTION: Faster coordination

2. INDEPENDENT: MultiSynapseNetwork with use_correlated_sampling=False  
   - Same quantum physics (same T₂, same dimer formation)
   - NO correlated sampling - each synapse evaluated independently
   - PREDICTION: Slower coordination (no entanglement benefit)

3. CLASSICAL: Exponential eligibility trace τ=100s
   - No quantum physics
   - Independent exponential traces matching T₂ timescale
   - PREDICTION: Similar to Independent (proves T₂ alone insufficient)

Task:
- N synapses must ALL commit to correct pattern
- Only global scalar reward (no per-synapse gradients)
- Protocol: stimulate → delay → reward → check commitment

Author: Sarah Davidson
University of Florida
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ScalingResult:
    """Results for one condition across all N values"""
    condition_type: str  # 'entangled', 'independent', 'classical'
    n_values: List[int] = field(default_factory=list)
    
    # Trials to converge (mean, std across trials)
    trials_mean: List[float] = field(default_factory=list)
    trials_std: List[float] = field(default_factory=list)
    
    # Success rate
    success_rates: List[float] = field(default_factory=list)
    
    # Fitted scaling
    scaling_exponent: float = 0.0
    scaling_r_squared: float = 0.0
    is_polynomial: bool = False


@dataclass
class NetworkCommunicationResult:
    """Complete experiment results"""
    entangled: ScalingResult = None
    independent: ScalingResult = None
    classical: ScalingResult = None
    
    # Comparison
    coordination_advantage_confirmed: bool = False
    entangled_vs_independent_ratio: float = 0.0
    entangled_vs_classical_ratio: float = 0.0
    
    timestamp: str = ""
    runtime_s: float = 0.0


# =============================================================================
# CLASSICAL BASELINE
# =============================================================================

class ClassicalEligibilityNetwork:
    """
    Classical network with exponential eligibility traces (τ=100s).
    No quantum physics - just exponential decay matching T₂ timescale.
    """
    
    def __init__(self, n_synapses: int, tau: float = 100.0):
        self.n_synapses = n_synapses
        self.tau = tau
        
        self.eligibility = np.zeros(n_synapses)
        self.committed = np.zeros(n_synapses, dtype=bool)
        self.time = 0.0
    
    def reset(self):
        self.eligibility = np.zeros(self.n_synapses)
        self.committed = np.zeros(self.n_synapses, dtype=bool)
        self.time = 0.0
    
    def stimulate(self, pattern: np.ndarray, duration: float, dt: float = 0.1):
        """Stimulate synapses - eligibility rises for active ones"""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            active = pattern > 0.5
            # Eligibility rises for active, decays for inactive
            self.eligibility[active] += dt * (1.0 - self.eligibility[active]) / 1.0
            self.eligibility *= np.exp(-dt / self.tau)
            self.time += dt
    
    def delay(self, duration: float, dt: float = 0.1):
        """Delay - all eligibilities decay"""
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.eligibility *= np.exp(-dt / self.tau)
            self.time += dt
    
    def apply_reward(self, threshold: float = 0.3):
        """Apply reward - commit synapses above threshold"""
        for i in range(self.n_synapses):
            if not self.committed[i] and self.eligibility[i] > threshold:
                self.committed[i] = True
    
    @property
    def all_committed(self) -> bool:
        return np.all(self.committed)


# =============================================================================
# COORDINATION TASK PROTOCOL
# =============================================================================

def run_quantum_coordination_trial(
    network: MultiSynapseNetwork,
    target_pattern: np.ndarray,
    stim_duration: float = 2.0,
    delay_duration: float = 5.0,
    reward_duration: float = 1.0,
    dt: float = 0.01,
    max_attempts: int = 50,
    verbose: bool = False
) -> Tuple[int, bool]:
    """
    Run coordination task with quantum network.
    
    Protocol per attempt:
    1. Reset network
    2. Stimulate synapses according to candidate pattern
    3. Delay (eligibility persists via quantum coherence)
    4. If candidate matches target, apply reward (triggers gate evaluation)
    5. Check if all synapses committed correctly
    
    The key difference between entangled/independent is in step 4:
    - Entangled: sample_correlated_eligibilities() → coordinated commitment
    - Independent: each synapse evaluated independently → uncoordinated
    """
    n_synapses = network.n_synapses
    
    for attempt in range(max_attempts):
        # Reset network
        network.reset()
        
        # Generate candidate pattern (starts random, then uses network state)
        if attempt < 5:
            candidate = np.random.choice([0, 1], size=n_synapses)
        else:
            # Use commitment state to guide exploration
            commit_probs = np.array([
                0.5 + 0.4 * (1 if getattr(s, '_camkii_committed', False) else 0)
                for s in network.synapses
            ])
            candidate = (np.random.random(n_synapses) < commit_probs).astype(int)
        
        # Phase 1: Stimulate according to candidate pattern
        # Step individual synapses with per-synapse voltage
        stim_steps = int(stim_duration / dt)
        for step in range(stim_steps):
            for i, syn in enumerate(network.synapses):
                voltage = -10e-3 if candidate[i] == 1 else -70e-3
                syn.step(dt, {'voltage': voltage, 'reward': False})
            
            # Update entanglement tracking after each step
            # This builds entanglement between co-activated synapses
            if step % 10 == 0:
                network.entanglement_tracker.collect_dimers(network.synapses, network.positions)
                network.entanglement_tracker.step(dt * 10, network.synapses, network.positions)
        
        # Phase 2: Delay
        delay_steps = int(delay_duration / dt)
        for step in range(delay_steps):
            network.step(dt, {'voltage': -70e-3, 'reward': False})
        
        # Phase 3: Reward if pattern matches
        pattern_correct = np.all(candidate == target_pattern)
        
        if pattern_correct:
            # Apply reward - this triggers coordinated/independent gate evaluation
            reward_steps = int(reward_duration / dt)
            for step in range(reward_steps):
                network.step(dt, {'voltage': -70e-3, 'reward': True})
            
            # Diagnostic: check correlation matrix and sampled eligibilities
            if verbose:
                C = network.entanglement_tracker.get_synapse_correlation_matrix(network.synapses)
                mean_corr = np.mean(C[np.triu_indices(n_synapses, k=1)])
                eligibilities = [s.get_eligibility() for s in network.synapses]
                print(f"      Correlation matrix mean (off-diag): {mean_corr:.3f}")
                print(f"      Eligibilities: {[f'{e:.2f}' for e in eligibilities]}")
        
        # Check commitment
        committed = np.array([
            getattr(s, '_camkii_committed', False) for s in network.synapses
        ])
        
        if verbose and attempt % 5 == 0:
            n_committed = np.sum(committed)
            print(f"    Attempt {attempt}: committed={n_committed}/{n_synapses}, "
                  f"pattern_correct={pattern_correct}")
        
        # Success: all committed
        if np.all(committed):
            return attempt + 1, True
    
    return max_attempts, False


def run_classical_coordination_trial(
    network: ClassicalEligibilityNetwork,
    target_pattern: np.ndarray,
    stim_duration: float = 2.0,
    delay_duration: float = 5.0,
    dt: float = 0.1,
    max_attempts: int = 50,
    verbose: bool = False
) -> Tuple[int, bool]:
    """Run coordination task with classical network."""
    n_synapses = network.n_synapses
    
    for attempt in range(max_attempts):
        network.reset()
        
        # Generate candidate
        if attempt < 5:
            candidate = np.random.choice([0, 1], size=n_synapses)
        else:
            commit_probs = 0.5 + 0.4 * network.committed.astype(float)
            candidate = (np.random.random(n_synapses) < commit_probs).astype(int)
        
        # Stimulate
        network.stimulate(candidate, stim_duration, dt=dt)
        
        # Delay
        network.delay(delay_duration, dt=dt)

        # Reward if correct
        if np.all(candidate == target_pattern):
            network.apply_reward()
        
        if verbose and attempt % 5 == 0:
            n_committed = np.sum(network.committed)
            print(f"    Attempt {attempt}: committed={n_committed}/{n_synapses}")
        
        if network.all_committed:
            return attempt + 1, True
    
    return max_attempts, False


# =============================================================================
# EXPERIMENT CLASS
# =============================================================================

class NetworkCommunicationExperiment:
    """
    Tests coordination via entanglement vs independent vs classical.
    Uses ACTUAL Model 6 physics.
    """
    
    def __init__(self, quick_mode: bool = False, validation_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.validation_mode = validation_mode
        self.verbose = verbose
        
        if validation_mode:
            # Ultra-fast: verify it runs in ~2-5 minutes
            self.n_values = [3, 4]
            self.n_trials = 3
            self.max_attempts = 15
            self.stim_duration = 0.1
            self.delay_duration = 2.0
            self.reward_duration = 0.2
            self.dt = 0.1  # 100ms steps - much faster
        elif quick_mode:
            self.n_values = [2, 3, 4]
            self.n_trials = 3
            self.max_attempts = 15
            self.stim_duration = 0.5
            self.delay_duration = 1.0
            self.reward_duration = 0.2
            self.dt = 0.02  # 20ms steps
        else:
            self.n_values = [2, 3, 4, 5, 6]
            self.n_trials = 10
            self.max_attempts = 50
            self.stim_duration = 2.0
            self.delay_duration = 5.0
            self.reward_duration = 1.0
            self.dt = 0.01  # 10ms steps
    
    def _create_quantum_network(self, n_synapses: int, 
                                 use_correlated: bool = True) -> MultiSynapseNetwork:
        """Create quantum network with specified coordination mode"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.environment.fraction_P31 = 1.0  # P31 for long T₂
        
        network = MultiSynapseNetwork(
            n_synapses=n_synapses,
            spacing_um=1.0,
            pattern='clustered',
            use_correlated_sampling=use_correlated  # KEY PARAMETER
        )
        
        print(f"  Created network: n={n_synapses}, use_correlated_sampling={use_correlated}")

        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        # Disable network-level field commitment for this experiment
        # We want to test individual synapse coordination via _evaluate_coordinated_gate
        network.field_threshold_kT = float('inf')  # Effectively disable
        
        return network
    
    def _run_scaling_sweep(self, condition_type: str) -> ScalingResult:
        """Run scaling sweep for one condition type"""
        result = ScalingResult(condition_type=condition_type)
        
        if self.verbose:
            print(f"\n--- {condition_type.upper()} ---")
        
        for n in self.n_values:
            trials_to_converge = []
            successes = 0
            
            if self.verbose:
                print(f"  N={n}: ", end='', flush=True)
            
            for trial in range(self.n_trials):
                # Generate random target
                np.random.seed(trial * 1000 + n)
                target = np.random.choice([0, 1], size=n)
                
                # Create network
                if condition_type == 'entangled':
                    network = self._create_quantum_network(n, use_correlated=True)
                    attempts, converged = run_quantum_coordination_trial(
                        network, target, 
                        self.stim_duration, self.delay_duration,
                        reward_duration=self.reward_duration,
                        dt=self.dt,
                        max_attempts=self.max_attempts
                    )
                elif condition_type == 'independent':
                    network = self._create_quantum_network(n, use_correlated=False)
                    attempts, converged = run_quantum_coordination_trial(
                        network, target,
                        self.stim_duration, self.delay_duration,
                        reward_duration=self.reward_duration,
                        dt=self.dt,
                        max_attempts=self.max_attempts
                    )
                else:  # classical
                    network = ClassicalEligibilityNetwork(n, tau=100.0)
                    attempts, converged = run_classical_coordination_trial(
                        network, target,
                        self.stim_duration, self.delay_duration,
                        dt=self.dt,
                        max_attempts=self.max_attempts
                    )
                
                trials_to_converge.append(attempts)
                if converged:
                    successes += 1
                
                if self.verbose:
                    print("." if converged else "x", end='', flush=True)
            
            # Record stats
            result.n_values.append(n)
            result.trials_mean.append(np.mean(trials_to_converge))
            result.trials_std.append(np.std(trials_to_converge))
            result.success_rates.append(successes / self.n_trials)
            
            if self.verbose:
                print(f" mean={np.mean(trials_to_converge):.1f}±{np.std(trials_to_converge):.1f}, "
                      f"success={successes}/{self.n_trials}")
        
        # Fit scaling
        self._fit_scaling(result)
        
        return result
    
    def _fit_scaling(self, result: ScalingResult):
        """Fit power law to scaling data"""
        if len(result.n_values) < 2:
            return
        
        # Only use points where we have successful convergence
        valid_mask = np.array(result.success_rates) > 0.5
        if np.sum(valid_mask) < 2:
            return
        
        n_valid = np.array(result.n_values)[valid_mask]
        trials_valid = np.array(result.trials_mean)[valid_mask]
        
        # Log-log fit
        log_n = np.log(n_valid)
        log_trials = np.log(trials_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_trials)
        
        result.scaling_exponent = slope
        result.scaling_r_squared = r_value ** 2
        result.is_polynomial = slope < 3.0
    
    def run(self) -> NetworkCommunicationResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        result = NetworkCommunicationResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        print("\n" + "=" * 70)
        print("NETWORK COMMUNICATION EXPERIMENT (Model 6)")
        print("=" * 70)
        print(f"\nTesting coordination via entanglement vs independent vs classical")
        print(f"N values: {self.n_values}")
        print(f"Trials per condition: {self.n_trials}")
        
        # Run all conditions
        result.entangled = self._run_scaling_sweep('entangled')
        result.independent = self._run_scaling_sweep('independent')
        result.classical = self._run_scaling_sweep('classical')
        
        # Compare
        if result.entangled.trials_mean and result.independent.trials_mean:
            ent_trials = result.entangled.trials_mean[-1]
            ind_trials = result.independent.trials_mean[-1]
            cls_trials = result.classical.trials_mean[-1]
            
            if ent_trials > 0:
                result.entangled_vs_independent_ratio = ind_trials / ent_trials
                result.entangled_vs_classical_ratio = cls_trials / ent_trials
            
            result.coordination_advantage_confirmed = (
                result.entangled_vs_independent_ratio > 1.5 or
                result.entangled.success_rates[-1] > result.independent.success_rates[-1] + 0.2
            )
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def print_summary(self, result: NetworkCommunicationResult):
        """Print formatted summary"""
        print("\n" + "=" * 70)
        print("NETWORK COMMUNICATION RESULTS")
        print("=" * 70)
        
        print("\n--- SCALING ANALYSIS ---")
        print(f"{'Condition':<15} {'Exponent':<12} {'R²':<8} {'Polynomial?':<12}")
        print("-" * 50)
        
        for name, res in [('Entangled', result.entangled), 
                          ('Independent', result.independent),
                          ('Classical', result.classical)]:
            if res:
                poly = "YES ✓" if res.is_polynomial else "NO ✗"
                print(f"{name:<15} {res.scaling_exponent:<12.2f} "
                      f"{res.scaling_r_squared:<8.3f} {poly:<12}")
        
        print("\n--- TRIALS TO CONVERGE ---")
        print(f"{'N':<6}", end='')
        for name in ['Entangled', 'Independent', 'Classical']:
            print(f"{name:<18}", end='')
        print()
        print("-" * 60)
        
        for i, n in enumerate(result.entangled.n_values if result.entangled else []):
            print(f"{n:<6}", end='')
            for res in [result.entangled, result.independent, result.classical]:
                if res and i < len(res.trials_mean):
                    print(f"{res.trials_mean[i]:.1f}±{res.trials_std[i]:.1f}".ljust(18), end='')
                else:
                    print("-".ljust(18), end='')
            print()
        
        print("\n--- COMPARISON ---")
        print(f"  Entangled vs Independent ratio: {result.entangled_vs_independent_ratio:.1f}x")
        print(f"  Entangled vs Classical ratio: {result.entangled_vs_classical_ratio:.1f}x")
        
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        
        if result.coordination_advantage_confirmed:
            print("""
✓ COORDINATION ADVANTAGE CONFIRMED

Entanglement provides coordination that classical mechanisms cannot match:
- Entangled: use_correlated_sampling=True → coordinated commitment
- Independent: use_correlated_sampling=False → uncoordinated (same T₂!)
- Classical: τ=100s exponential → no quantum benefit

This demonstrates Gap 4: Coordination via entanglement.
""")
        else:
            print("""
⚠ Results inconclusive - check experimental parameters

Try:
- Larger N values
- Longer delays
- More trials
""")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: NetworkCommunicationResult, 
             output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        colors = {
            'entangled': '#2E86AB',
            'independent': '#E94F37',
            'classical': '#8B8B8B'
        }
        
        # Panel A: Trials vs N
        ax = axes[0]
        for name, res, marker in [('entangled', result.entangled, 'o'),
                                   ('independent', result.independent, 's'),
                                   ('classical', result.classical, '^')]:
            if res and res.n_values:
                ax.errorbar(res.n_values, res.trials_mean, yerr=res.trials_std,
                           color=colors[name], marker=marker, markersize=8,
                           linewidth=2, capsize=5, label=name.capitalize())
        
        ax.set_xlabel('Number of Synapses (N)')
        ax.set_ylabel('Trials to Converge')
        ax.set_title('A. Learning Speed\n(lower = faster)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel B: Success Rate
        ax = axes[1]
        n_vals = result.entangled.n_values if result.entangled else []
        x = np.arange(len(n_vals))
        width = 0.25
        
        for i, (name, res) in enumerate([('entangled', result.entangled),
                                          ('independent', result.independent),
                                          ('classical', result.classical)]):
            if res and res.success_rates:
                ax.bar(x + i*width, res.success_rates, width, 
                       color=colors[name], label=name.capitalize())
        
        ax.set_xlabel('Network Size (N)')
        ax.set_ylabel('Success Rate')
        ax.set_title('B. Convergence Success Rate')
        ax.set_xticks(x + width)
        ax.set_xticklabels(n_vals)
        ax.legend()
        ax.set_ylim(0, 1.15)
        
        # Panel C: Speedup Ratio
        ax = axes[2]
        
        speedup_ind = []
        speedup_cls = []
        for i in range(len(n_vals)):
            ent = result.entangled.trials_mean[i] if result.entangled else 1
            ind = result.independent.trials_mean[i] if result.independent else 1
            cls = result.classical.trials_mean[i] if result.classical else 1
            speedup_ind.append(ind / ent if ent > 0 else 1)
            speedup_cls.append(cls / ent if ent > 0 else 1)
        
        x = np.arange(len(n_vals))
        width = 0.35
        ax.bar(x - width/2, speedup_ind, width, color=colors['independent'], 
               label='vs Independent', alpha=0.8)
        ax.bar(x + width/2, speedup_cls, width, color=colors['classical'],
               label='vs Classical', alpha=0.8)
        
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Network Size (N)')
        ax.set_ylabel('Speedup Ratio')
        ax.set_title('C. Entanglement Advantage')
        ax.set_xticks(x)
        ax.set_xticklabels(n_vals)
        ax.legend()
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / 'network_communication.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_results(self, result: NetworkCommunicationResult, path: Path):
        """Save results to JSON"""
        def scaling_to_dict(s: ScalingResult) -> dict:
            if s is None:
                return None
            return {
                'condition_type': s.condition_type,
                'n_values': s.n_values,
                'trials_mean': s.trials_mean,
                'trials_std': s.trials_std,
                'success_rates': s.success_rates,
                'scaling_exponent': s.scaling_exponent,
                'scaling_r_squared': s.scaling_r_squared,
                'is_polynomial': str(s.is_polynomial)
            }
        
        data = {
            'experiment': 'network_communication',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'entangled': scaling_to_dict(result.entangled),
            'independent': scaling_to_dict(result.independent),
            'classical': scaling_to_dict(result.classical),
            'coordination_advantage_confirmed': str(result.coordination_advantage_confirmed),
            'entangled_vs_independent_ratio': result.entangled_vs_independent_ratio,
            'entangled_vs_classical_ratio': result.entangled_vs_classical_ratio,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary_dict(self, result: NetworkCommunicationResult) -> dict:
        """Get summary for master results"""
        return {
            'coordination_advantage': result.coordination_advantage_confirmed,
            'speedup_vs_independent': result.entangled_vs_independent_ratio,
            'speedup_vs_classical': result.entangled_vs_classical_ratio,
            'entangled_success_rate': result.entangled.success_rates[-1] if result.entangled else 0,
            'independent_success_rate': result.independent.success_rates[-1] if result.independent else 0,
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--validation', action='store_true', help='Ultra-fast validation mode')
    parser.add_argument('--output', type=str, default='results', help='Output dir')
    args = parser.parse_args()
    
    exp = NetworkCommunicationExperiment(quick_mode=args.quick, validation_mode=args.validation )
    result = exp.run()
    exp.print_summary(result)
    exp.plot(result, output_dir=Path(args.output))
    exp.save_results(result, Path(args.output) / 'network_communication.json')
    
    plt.show()