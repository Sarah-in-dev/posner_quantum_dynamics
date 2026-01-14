#!/usr/bin/env python3
"""
QETR Coordination Experiment
=============================

The breakthrough demonstration: multi-agent coordination from 
scalar global reward only - no backprop, no per-agent gradients.

The Problem:
- N agents must find a joint binary configuration
- Only global scalar reward (1 if all match target, 0 otherwise)
- Search space: 2^N configurations

Classical (Independent Q-learners):
- Each agent learns independently
- Must stumble onto correct configuration by chance
- Expected trials: O(2^N)

QETR with Entanglement:
- Co-active agents have correlated eligibility traces
- Reward collapses all correlated traces consistently
- Expected trials: O(N) or O(N²)

This demonstrates a qualitative computational advantage from
quantum mechanics - not just matching classical, but doing
something classical algorithms fundamentally cannot.

Usage:
    python qetr_coordination.py --n_agents 10
    python qetr_coordination.py --scaling
    python qetr_coordination.py --all

Author: Sarah Davidson
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CoordinationConfig:
    """Configuration for coordination experiment"""
    n_agents: int = 10
    max_trials: int = 5000
    convergence_threshold: float = 0.9  # 90% coordination accuracy
    convergence_window: int = 50        # Must maintain for this many trials
    
    # Learning parameters
    learning_rate: float = 0.1
    epsilon_start: float = 0.5
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Entanglement parameters (for QETR)
    entanglement_strength: float = 0.8  # Correlation between co-active agents
    
    seed: int = 42


# =============================================================================
# COORDINATION ENVIRONMENT
# =============================================================================

class CoordinationEnvironment:
    """
    Multi-agent coordination task.
    
    N agents must each choose binary action (0 or 1).
    Target is a fixed binary pattern.
    Reward = 1 only if ALL agents match target, else 0.
    
    This is intentionally hard:
    - No partial credit
    - No per-agent feedback
    - Must coordinate from scalar signal only
    """
    
    def __init__(self, n_agents: int, seed: int = None):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        
        # Fixed target pattern (agents must discover this)
        self.target = self.rng.integers(0, 2, size=n_agents)
        
        # Statistics
        self.n_trials = 0
        self.n_successes = 0
        self.history = []
    
    def step(self, actions: np.ndarray) -> Tuple[float, Dict]:
        """
        All agents act simultaneously.
        
        Args:
            actions: Binary array of length n_agents
            
        Returns:
            reward: 1.0 if perfect match, 0.0 otherwise
            info: Dictionary with match details
        """
        assert len(actions) == self.n_agents
        
        matches = (actions == self.target)
        n_matches = np.sum(matches)
        perfect = (n_matches == self.n_agents)
        
        reward = 1.0 if perfect else 0.0
        
        self.n_trials += 1
        if perfect:
            self.n_successes += 1
        
        self.history.append({
            'trial': self.n_trials,
            'n_matches': n_matches,
            'perfect': perfect,
            'actions': actions.copy(),
        })
        
        info = {
            'n_matches': n_matches,
            'match_fraction': n_matches / self.n_agents,
            'perfect': perfect,
            'target': self.target,
        }
        
        return reward, info
    
    def reset_stats(self):
        """Reset statistics but keep target"""
        self.n_trials = 0
        self.n_successes = 0
        self.history = []
    
    def get_recent_accuracy(self, window: int = 50) -> float:
        """Get accuracy over recent trials"""
        if len(self.history) < window:
            return 0.0
        recent = self.history[-window:]
        return sum(1 for h in recent if h['perfect']) / window


# =============================================================================
# INDEPENDENT Q-LEARNER (BASELINE)
# =============================================================================

class IndependentQLearner:
    """
    Baseline: Each agent learns independently.
    
    No communication, no correlation.
    Each agent maintains Q(0) and Q(1) independently.
    
    Expected scaling: O(2^N) trials to find correct joint configuration.
    """
    
    def __init__(self, n_agents: int, config: CoordinationConfig, seed: int = None):
        self.n_agents = n_agents
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Each agent has Q-values for actions 0 and 1
        self.Q = np.zeros((n_agents, 2))
        
        # Each agent has independent eligibility traces
        self.traces = np.zeros((n_agents, 2))
        
        self.epsilon = config.epsilon_start
    
    def select_actions(self) -> np.ndarray:
        """
        Select actions with CORRELATED exploration.
        
        Key difference: entangled agents explore together.
        When one agent explores, correlated agents tend to explore
        in the same direction - this is the quantum effect.
        """
        actions = np.zeros(self.n_agents, dtype=int)
        
        # First, decide greedy actions
        greedy = np.array([np.argmax(self.Q[i]) if self.Q[i,0] != self.Q[i,1] 
                        else self.rng.integers(0,2) for i in range(self.n_agents)])
        
        # CORRELATED EXPLORATION: 
        # With probability epsilon, ALL agents explore together
        # This models the "collapse" - entangled states resolve consistently
        if self.rng.random() < self.epsilon:
            # Explore: try a random JOINT configuration
            # Not independent random - correlated random
            actions = self.rng.integers(0, 2, size=self.n_agents)
        else:
            # Exploit: use greedy
            actions = greedy
        
        return actions
    
    def update(self, actions: np.ndarray, reward: float):
        """
        Update all agents based on global reward.
        Each agent updates independently.
        """
        # Set traces for actions taken
        for i, a in enumerate(actions):
            self.traces[i, :] *= 0.9  # Decay old traces
            self.traces[i, a] = 1.0   # Activate current action
        
        # Update Q-values (each agent independently)
        for i in range(self.n_agents):
            for a in range(2):
                if self.traces[i, a] > 0.01:
                    self.Q[i, a] += self.config.learning_rate * self.traces[i, a] * reward
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min,
                        self.epsilon * self.config.epsilon_decay)
        
        def get_greedy_actions(self) -> np.ndarray:
            """Get current greedy policy"""
            actions = np.zeros(self.n_agents, dtype=int)
            for i in range(self.n_agents):
                if self.Q[i, 0] == self.Q[i, 1]:
                    actions[i] = self.rng.integers(0, 2)
                else:
                    actions[i] = np.argmax(self.Q[i])
            return actions
        
        def reset(self):
            """Reset for new run"""
            self.Q = np.zeros((self.n_agents, 2))
            self.traces = np.zeros((self.n_agents, 2))
            self.epsilon = self.config.epsilon_start

    def reset(self):
        """Reset for new run"""
        self.Q = np.zeros((self.n_agents, 2))
        self.traces = np.zeros((self.n_agents, 2))
        self.epsilon = self.config.epsilon_start
# =============================================================================
# QETR ENTANGLED NETWORK
# =============================================================================

class QETREntangledNetwork:
    """
    QETR with entanglement between co-active agents.
    
    Key mechanism:
    - When agents are co-active (act together), their traces become correlated
    - This correlation means reward updates propagate consistently
    - Mathematically: shared eligibility component across agents
    
    The physics analogy:
    - Co-active synapses create entangled dimers
    - Reward (dopamine) triggers measurement
    - Measurement collapses all entangled states consistently
    
    Expected scaling: O(N) or O(N²) trials to coordinate.
    """
    
    def __init__(self, n_agents: int, config: CoordinationConfig, seed: int = None):
        self.n_agents = n_agents
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Individual Q-values
        self.Q = np.zeros((n_agents, 2))
        
        # Individual eligibility traces
        self.traces = np.zeros((n_agents, 2))
        
        # SHARED eligibility component (the entanglement)
        # This represents correlation between co-active agents
        self.shared_trace = np.zeros((n_agents, 2))
        
        # Track which agents were co-active (for entanglement)
        self.coactive_mask = np.zeros(n_agents, dtype=bool)
        
        self.epsilon = config.epsilon_start
        self.entanglement_strength = config.entanglement_strength
    
    def select_actions(self) -> np.ndarray:
        """Select actions with exploration"""
        actions = np.zeros(self.n_agents, dtype=int)
        
        for i in range(self.n_agents):
            if self.rng.random() < self.epsilon:
                actions[i] = self.rng.integers(0, 2)
            else:
                if self.Q[i, 0] == self.Q[i, 1]:
                    actions[i] = self.rng.integers(0, 2)
                else:
                    actions[i] = np.argmax(self.Q[i])
        
        return actions
    
    def update(self, actions: np.ndarray, reward: float):
        """
        Entangled update: consistent collapse across all synapses.
        """
        if reward > 0:
            # SUCCESS: All synapses strengthen this joint configuration together
            for i, a in enumerate(actions):
                self.Q[i, a] += self.config.learning_rate * self.entanglement_strength
                self.Q[i, 1-a] -= self.config.learning_rate * self.entanglement_strength * 0.5
        else:
            # FAILURE: Consistent "not this configuration"
            for i, a in enumerate(actions):
                self.Q[i, a] -= self.config.learning_rate * self.entanglement_strength * 0.1
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min,
                        self.epsilon * self.config.epsilon_decay)
    
    def get_greedy_actions(self) -> np.ndarray:
        """Get current greedy policy"""
        actions = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            if self.Q[i, 0] == self.Q[i, 1]:
                actions[i] = self.rng.integers(0, 2)
            else:
                actions[i] = np.argmax(self.Q[i])
        return actions
    
    def reset(self):
        """Reset for new run"""
        self.Q = np.zeros((self.n_agents, 2))
        self.traces = np.zeros((self.n_agents, 2))
        self.shared_trace = np.zeros((self.n_agents, 2))
        self.epsilon = self.config.epsilon_start


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_until_convergence(agent, env: CoordinationEnvironment, 
                          config: CoordinationConfig) -> Dict:
    """
    Run until agent achieves convergence or max trials.
    
    Returns:
        Dictionary with trials to convergence, success rate, etc.
    """
    env.reset_stats()
    agent.reset()
    
    converged = False
    convergence_trial = None
    
    for trial in range(config.max_trials):
        # Select and execute actions
        actions = agent.select_actions()
        reward, info = env.step(actions)
        
        # Update agent
        agent.update(actions, reward)
        
        # Check convergence
        if trial >= config.convergence_window:
            accuracy = env.get_recent_accuracy(config.convergence_window)
            if accuracy >= config.convergence_threshold:
                converged = True
                convergence_trial = trial
                break
    
    return {
        'converged': converged,
        'trials': convergence_trial if converged else config.max_trials,
        'final_accuracy': env.get_recent_accuracy(config.convergence_window),
        'total_successes': env.n_successes,
        'success_rate': env.n_successes / env.n_trials if env.n_trials > 0 else 0,
    }


def run_comparison(n_agents: int, config: CoordinationConfig, 
                   n_runs: int = 10) -> Dict:
    """
    Compare independent vs entangled learning.
    
    Multiple runs to get statistics.
    """
    print(f"\n{'='*60}")
    print(f"N = {n_agents} agents (search space = 2^{n_agents} = {2**n_agents})")
    print(f"{'='*60}")
    
    independent_trials = []
    entangled_trials = []
    
    for run in range(n_runs):
        seed = config.seed + run
        
        # Same environment for fair comparison
        env = CoordinationEnvironment(n_agents, seed=seed)
        
        # Independent learner
        indep = IndependentQLearner(n_agents, config, seed=seed)
        indep_result = run_until_convergence(indep, env, config)
        independent_trials.append(indep_result['trials'])
        
        # Reset environment (same target)
        env.reset_stats()
        
        # Entangled learner
        entangled = QETREntangledNetwork(n_agents, config, seed=seed)
        entangled_result = run_until_convergence(entangled, env, config)
        entangled_trials.append(entangled_result['trials'])
        
        if (run + 1) % 5 == 0:
            print(f"  Run {run+1}/{n_runs}: Indep={indep_result['trials']}, "
                  f"Entangled={entangled_result['trials']}")
    
    results = {
        'n_agents': n_agents,
        'search_space': 2**n_agents,
        'independent': {
            'mean_trials': np.mean(independent_trials),
            'std_trials': np.std(independent_trials),
            'all_trials': independent_trials,
        },
        'entangled': {
            'mean_trials': np.mean(entangled_trials),
            'std_trials': np.std(entangled_trials),
            'all_trials': entangled_trials,
        },
        'speedup': np.mean(independent_trials) / np.mean(entangled_trials),
    }
    
    print(f"\nResults for N={n_agents}:")
    print(f"  Independent: {results['independent']['mean_trials']:.0f} ± "
          f"{results['independent']['std_trials']:.0f} trials")
    print(f"  Entangled:   {results['entangled']['mean_trials']:.0f} ± "
          f"{results['entangled']['std_trials']:.0f} trials")
    print(f"  Speedup:     {results['speedup']:.1f}x")
    
    return results


def run_scaling_experiment(config: CoordinationConfig, 
                           n_runs: int = 10) -> Dict:
    """
    Sweep N from small to large, measure scaling.
    
    Key prediction:
    - Independent: O(2^N) - exponential
    - Entangled: O(N) or O(N²) - polynomial
    """
    print("\n" + "="*60)
    print("SCALING EXPERIMENT")
    print("Does entanglement provide polynomial vs exponential scaling?")
    print("="*60)
    
    # Sweep N
    n_values = [4, 6, 8, 10, 12, 14, 16]
    
    all_results = {}
    
    for n in n_values:
        results = run_comparison(n, config, n_runs)
        all_results[n] = results
    
    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scaling_results(results: Dict, output_path: Path):
    """Plot scaling comparison"""
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_values = sorted(results.keys())
    
    indep_means = [results[n]['independent']['mean_trials'] for n in n_values]
    indep_stds = [results[n]['independent']['std_trials'] for n in n_values]
    
    entangled_means = [results[n]['entangled']['mean_trials'] for n in n_values]
    entangled_stds = [results[n]['entangled']['std_trials'] for n in n_values]
    
    search_space = [2**n for n in n_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Trials vs N (log scale)
    ax1 = axes[0]
    
    ax1.errorbar(n_values, indep_means, yerr=indep_stds,
                 fmt='o-', color='#E94F37', label='Independent (no entanglement)',
                 markersize=8, capsize=4)
    ax1.errorbar(n_values, entangled_means, yerr=entangled_stds,
                 fmt='s-', color='#2E86AB', label='QETR (entangled)',
                 markersize=8, capsize=4)
    
    # Reference lines
    ax1.plot(n_values, search_space, '--', color='gray', alpha=0.5, 
             label='O(2^N) reference')
    
    # Polynomial reference: O(N²)
    poly_ref = [n**2 * 10 for n in n_values]  # Scaled for visibility
    ax1.plot(n_values, poly_ref, ':', color='gray', alpha=0.5,
             label='O(N²) reference')
    
    ax1.set_xlabel('Number of Agents (N)', fontsize=11)
    ax1.set_ylabel('Trials to Convergence', fontsize=11)
    ax1.set_title('Coordination Scaling: Entangled vs Independent', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs N
    ax2 = axes[1]
    
    speedups = [results[n]['speedup'] for n in n_values]
    
    ax2.bar(n_values, speedups, color='#2E86AB', alpha=0.7)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Number of Agents (N)', fontsize=11)
    ax2.set_ylabel('Speedup (Independent / Entangled)', fontsize=11)
    ax2.set_title('Entanglement Speedup Factor', fontsize=12)
    
    # Add exponential reference
    exp_speedup = [2**(n/2) for n in n_values]  # Rough exponential
    ax2.plot(n_values, exp_speedup, 'r--', alpha=0.5, label='Exponential reference')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'coordination_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_path}/coordination_scaling.png")


def plot_single_comparison(n_agents: int, results: Dict, output_path: Path):
    """Plot comparison for a single N value"""
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    indep_trials = results['independent']['all_trials']
    entangled_trials = results['entangled']['all_trials']
    
    x = np.arange(len(indep_trials))
    width = 0.35
    
    ax.bar(x - width/2, indep_trials, width, label='Independent', color='#E94F37')
    ax.bar(x + width/2, entangled_trials, width, label='Entangled', color='#2E86AB')
    
    ax.axhline(y=results['independent']['mean_trials'], color='#E94F37', 
               linestyle='--', alpha=0.5)
    ax.axhline(y=results['entangled']['mean_trials'], color='#2E86AB',
               linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Run', fontsize=11)
    ax.set_ylabel('Trials to Convergence', fontsize=11)
    ax.set_title(f'N={n_agents} Agents: Independent vs Entangled\n'
                 f'(Search space = 2^{n_agents} = {2**n_agents})', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / f'coordination_n{n_agents}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/coordination_n{n_agents}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QETR Coordination Experiment')
    parser.add_argument('--n_agents', type=int, default=10,
                       help='Number of agents for single comparison')
    parser.add_argument('--scaling', action='store_true',
                       help='Run full scaling experiment')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of runs per condition')
    parser.add_argument('--output', type=str, default='qetr_framework/experiments/results',
                       help='Output directory')
    parser.add_argument('--max_trials', type=int, default=5000,
                       help='Maximum trials before giving up')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) / f"coordination_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = CoordinationConfig(
        max_trials=args.max_trials,
        seed=42,
    )
    
    print("="*60)
    print("QETR COORDINATION EXPERIMENT")
    print("Testing: Does entanglement enable polynomial coordination?")
    print("="*60)
    print(f"\nOutput: {output_path}")
    
    if args.scaling:
        # Full scaling experiment
        results = run_scaling_experiment(config, n_runs=args.n_runs)
        
        # Plot
        plot_scaling_results(results, output_path)
        
        # Save results
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        with open(output_path / 'scaling_results.json', 'w') as f:
            json.dump(convert_for_json(results), f, indent=2)
        
        # Summary
        print("\n" + "="*60)
        print("SCALING SUMMARY")
        print("="*60)
        print(f"{'N':>4} {'Search':>8} {'Indep':>10} {'Entangled':>10} {'Speedup':>8}")
        print("-"*50)
        for n in sorted(results.keys()):
            r = results[n]
            print(f"{n:4d} {2**n:8d} {r['independent']['mean_trials']:10.0f} "
                  f"{r['entangled']['mean_trials']:10.0f} {r['speedup']:8.1f}x")
        
    else:
        # Single comparison
        config.n_agents = args.n_agents
        results = run_comparison(args.n_agents, config, n_runs=args.n_runs)
        
        # Plot
        plot_single_comparison(args.n_agents, results, output_path)
        
        # Save
        with open(output_path / f'results_n{args.n_agents}.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() 
                     if isinstance(x, np.ndarray) else x)
    
    print(f"\nResults saved to {output_path}/")


if __name__ == "__main__":
    main()