#!/usr/bin/env python3
"""
QETR Scenario-Based Experiments
================================

Configurable scenarios to find where TD(λ) tuning matters
and QETR's physics-derived λ provides advantage.

The core question: Does λ = exp(-dt/T₂) from physics match 
or beat hand-tuned λ values?

Usage:
    python qetr_scenarios.py --scenario easy
    python qetr_scenarios.py --scenario hard_delay
    python qetr_scenarios.py --list
    python qetr_scenarios.py --all

Author: Sarah Davidson
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from datetime import datetime
import argparse


# =============================================================================
# SCENARIO DEFINITION
# =============================================================================

@dataclass
class Scenario:
    """Configuration for a single experimental scenario"""
    name: str
    description: str
    
    # Task parameters
    n_actions: int = 4
    delay_min: float = 30.0
    delay_max: float = 90.0
    
    # Training parameters
    n_episodes: int = 30
    n_trials_per_episode: int = 50
    dt: float = 0.1
    
    # Difficulty modifiers
    reward_noise: float = 0.0      # Stochastic reward (0 = deterministic)
    reward_magnitude: float = 1.0   # Scale of reward
    
    # Learning parameters
    learning_rate: float = 0.1
    epsilon_start: float = 0.3
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.01
    
    seed: int = 42


# Pre-defined scenarios
SCENARIOS = {
    "easy": Scenario(
        name="easy",
        description="Baseline: 4 actions, 30-90s delays",
        n_actions=4,
        delay_min=30.0,
        delay_max=90.0,
    ),
    
    "more_actions": Scenario(
        name="more_actions",
        description="More actions: 10 actions, 30-90s delays",
        n_actions=10,
        delay_min=30.0,
        delay_max=90.0,
    ),
    
    "many_actions": Scenario(
        name="many_actions",
        description="Many actions: 20 actions, 30-90s delays",
        n_actions=20,
        delay_min=30.0,
        delay_max=90.0,
    ),
    
    "long_delay": Scenario(
        name="long_delay",
        description="Longer delays: 4 actions, 60-120s delays",
        n_actions=4,
        delay_min=60.0,
        delay_max=120.0,
    ),
    
    "very_long_delay": Scenario(
        name="very_long_delay",
        description="Very long delays: 4 actions, 90-180s delays",
        n_actions=4,
        delay_min=90.0,
        delay_max=180.0,
    ),
    
    "hard_combined": Scenario(
        name="hard_combined",
        description="Hard: 10 actions, 60-120s delays",
        n_actions=10,
        delay_min=60.0,
        delay_max=120.0,
    ),
    
    "noisy_reward": Scenario(
        name="noisy_reward",
        description="Noisy: 4 actions, 30-90s delays, 30% reward noise",
        n_actions=4,
        delay_min=30.0,
        delay_max=90.0,
        reward_noise=0.3,
    ),
    
    "extreme": Scenario(
        name="extreme",
        description="Extreme: 15 actions, 90-180s delays, 20% noise",
        n_actions=15,
        delay_min=90.0,
        delay_max=180.0,
        reward_noise=0.2,
    ),
    
    "few_trials": Scenario(
        name="few_trials",
        description="Few trials: 4 actions, 30-90s, only 20 trials/episode",
        n_actions=4,
        delay_min=30.0,
        delay_max=90.0,
        n_trials_per_episode=20,
    ),
    
    "low_learning_rate": Scenario(
        name="low_learning_rate",
        description="Low LR: 4 actions, 30-90s, learning_rate=0.01",
        n_actions=4,
        delay_min=30.0,
        delay_max=90.0,
        learning_rate=0.01,
    ),
}


# =============================================================================
# PHYSICS CONSTANTS (from your Model 6)
# =============================================================================

@dataclass(frozen=True)
class PhysicsConstants:
    """Immutable physics parameters - NOT hyperparameters"""
    T2_P31: float = 216.0   # seconds (singlet lifetime)
    T2_P32: float = 0.4     # seconds
    
    def get_effective_lambda(self, isotope: str, dt: float) -> float:
        """λ_effective = exp(-dt/T₂) - derived from physics"""
        T2 = self.T2_P31 if isotope == "P31" else self.T2_P32
        return np.exp(-dt / T2)


PHYSICS = PhysicsConstants()


# =============================================================================
# SIMPLE ELIGIBILITY TRACE (standalone, no Model 6 dependency)
# =============================================================================

class EligibilityTrace:
    """
    Eligibility trace with physics-derived or manual decay.
    
    For QETR: decay = exp(-dt/T₂) where T₂ is from nuclear spin physics
    For TD(λ): decay = λ where λ is a hyperparameter
    """
    
    def __init__(self, decay_per_step: float):
        """
        Args:
            decay_per_step: Multiplicative decay per dt step
                           For QETR: exp(-dt/T₂)
                           For TD(λ): λ
        """
        self.decay = decay_per_step
        self.value = 0.0
    
    def activate(self, strength: float = 1.0):
        self.value = strength
    
    def step(self):
        self.value *= self.decay
    
    def reset(self):
        self.value = 0.0


# =============================================================================
# AGENTS
# =============================================================================

class BaseAgent:
    """Base class for RL agents"""
    
    def __init__(self, n_actions: int, config: Scenario, seed: int = None):
        self.n_actions = n_actions
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.Q = np.zeros(n_actions)
        self.traces = []  # Subclass fills this
        
        self.epsilon = config.epsilon_start
    
    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        
        # Break ties randomly
        max_val = np.max(self.Q)
        best = np.where(self.Q == max_val)[0]
        return self.rng.choice(best)
    
    def activate(self, action: int):
        self.traces[action].activate(1.0)
    
    def step_time(self):
        for trace in self.traces:
            trace.step()
    
    def update(self, reward: float):
        for i, trace in enumerate(self.traces):
            if trace.value > 0.001:
                self.Q[i] += self.config.learning_rate * trace.value * reward
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
    
    def reset(self, reset_weights: bool = False):
        for trace in self.traces:
            trace.reset()
        if reset_weights:
            self.Q = np.zeros(self.n_actions)
        self.epsilon = self.config.epsilon_start
    
    def get_predicted_action(self) -> int:
        max_val = np.max(self.Q)
        best = np.where(self.Q == max_val)[0]
        return self.rng.choice(best)


class QETRAgent(BaseAgent):
    """
    QETR Agent: λ derived from physics, NOT tuned.
    
    λ_effective = exp(-dt/T₂)
    """
    
    def __init__(self, n_actions: int, config: Scenario, 
                 isotope: str = "P31", seed: int = None):
        super().__init__(n_actions, config, seed)
        
        self.isotope = isotope
        self.T2 = PHYSICS.T2_P31 if isotope == "P31" else PHYSICS.T2_P32
        
        # λ from physics
        self.effective_lambda = PHYSICS.get_effective_lambda(isotope, config.dt)
        
        # Create traces with physics-derived decay
        self.traces = [EligibilityTrace(self.effective_lambda) 
                      for _ in range(n_actions)]
    
    def __repr__(self):
        return f"QETR-{self.isotope}(T₂={self.T2:.1f}s, λ={self.effective_lambda:.6f})"


class TDLambdaAgent(BaseAgent):
    """
    TD(λ) Agent: λ is a hyperparameter that must be tuned.
    """
    
    def __init__(self, n_actions: int, config: Scenario,
                 lambda_: float = 0.9, seed: int = None):
        super().__init__(n_actions, config, seed)
        
        self.lambda_ = lambda_
        
        # Create traces with manual λ
        self.traces = [EligibilityTrace(lambda_) for _ in range(n_actions)]
    
    def __repr__(self):
        return f"TD(λ={self.lambda_})"


# =============================================================================
# ENVIRONMENT
# =============================================================================

class DelayedRewardEnv:
    """Simple delayed reward environment"""
    
    def __init__(self, scenario: Scenario, seed: int = None):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        
        self.n_actions = scenario.n_actions
        self.correct_action = 0  # Fixed for consistency
        self.pending_reward = None
        self.time = 0.0
    
    def reset(self):
        self.pending_reward = None
        self.time = 0.0
        # Don't change correct_action - keep consistent across episodes
    
    def step(self, action: int) -> float:
        """Take action, schedule delayed reward if correct"""
        self.time += self.scenario.dt
        
        if action == self.correct_action:
            # Schedule reward with random delay
            delay = self.rng.uniform(self.scenario.delay_min, 
                                    self.scenario.delay_max)
            reward = self.scenario.reward_magnitude
            
            # Add noise if configured
            if self.scenario.reward_noise > 0:
                noise = self.rng.normal(0, self.scenario.reward_noise)
                reward = max(0, reward + noise)
            
            self.pending_reward = (self.time + delay, reward)
        
        return 0.0  # No immediate reward
    
    def step_time(self) -> float:
        """Advance time, return reward if due"""
        self.time += self.scenario.dt
        
        if self.pending_reward is not None:
            delivery_time, reward = self.pending_reward
            if self.time >= delivery_time:
                self.pending_reward = None
                return reward
        
        return 0.0


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_trial(agent: BaseAgent, env: DelayedRewardEnv) -> bool:
    """Run one trial, return whether agent predicts correct action"""
    
    # Select and take action
    action = agent.select_action()
    env.step(action)
    agent.activate(action)
    
    # Wait for delayed reward
    max_wait_steps = int(200.0 / env.scenario.dt)  # 200s max
    
    for _ in range(max_wait_steps):
        agent.step_time()
        reward = env.step_time()
        
        if reward > 0:
            agent.update(reward)
            break
    
    # Check if agent predicts correctly
    return agent.get_predicted_action() == env.correct_action


def run_episode(agent: BaseAgent, env: DelayedRewardEnv, 
                n_trials: int) -> float:
    """Run one episode, return accuracy"""
    env.reset()
    agent.reset(reset_weights=False)
    
    correct = 0
    for _ in range(n_trials):
        if run_single_trial(agent, env):
            correct += 1
    
    return correct / n_trials


def evaluate_agent(agent: BaseAgent, scenario: Scenario) -> Dict:
    """Evaluate an agent on a scenario"""
    env = DelayedRewardEnv(scenario, seed=scenario.seed)
    
    # Fresh agent for this evaluation
    agent.reset(reset_weights=True)
    
    accuracies = []
    for ep in range(scenario.n_episodes):
        acc = run_episode(agent, env, scenario.n_trials_per_episode)
        accuracies.append(acc)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'final_accuracy': accuracies[-1],
        'accuracies': accuracies,
    }


def run_td_comparison(scenario: Scenario) -> Dict:
    """
    Run TD(λ) comparison for a given scenario.
    
    This is the key experiment: compare physics-derived λ vs tuned λ.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.name}")
    print(f"{scenario.description}")
    print(f"  n_actions={scenario.n_actions}, delay={scenario.delay_min}-{scenario.delay_max}s")
    print(f"{'='*60}")
    
    results = {}
    
    # QETR-P31 (physics-derived, no tuning)
    qetr = QETRAgent(scenario.n_actions, scenario, isotope="P31", seed=scenario.seed)
    print(f"\n{qetr}")
    qetr_results = evaluate_agent(qetr, scenario)
    results['QETR-P31'] = {
        'accuracy': qetr_results['mean_accuracy'],
        'std': qetr_results['std_accuracy'],
        'T2': qetr.T2,
        'lambda': qetr.effective_lambda,
        'tuning_required': False,
        'accuracies': qetr_results['accuracies'],
    }
    print(f"  Accuracy: {qetr_results['mean_accuracy']:.1%} ± {qetr_results['std_accuracy']:.1%}")
    
    # TD(λ) with various λ values (hyperparameter search)
    lambda_values = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999]
    
    print(f"\nTD(λ) hyperparameter sweep:")
    best_lambda = None
    best_accuracy = 0.0
    
    for lam in lambda_values:
        td_agent = TDLambdaAgent(scenario.n_actions, scenario, 
                                  lambda_=lam, seed=scenario.seed)
        td_results = evaluate_agent(td_agent, scenario)
        
        results[f'TD(λ={lam})'] = {
            'accuracy': td_results['mean_accuracy'],
            'std': td_results['std_accuracy'],
            'lambda': lam,
            'tuning_required': True,
            'accuracies': td_results['accuracies'],
        }
        
        marker = ""
        if td_results['mean_accuracy'] > best_accuracy:
            best_accuracy = td_results['mean_accuracy']
            best_lambda = lam
            marker = " ← best so far"
        
        print(f"  λ={lam}: {td_results['mean_accuracy']:.1%} ± {td_results['std_accuracy']:.1%}{marker}")
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"QETR-P31 (no tuning):  {results['QETR-P31']['accuracy']:.1%}")
    print(f"Best TD(λ={best_lambda}):    {best_accuracy:.1%}")
    
    qetr_acc = results['QETR-P31']['accuracy']
    if qetr_acc >= best_accuracy - 0.02:
        print(f"✓ Physics-derived λ matches or beats tuned baseline")
    else:
        print(f"✗ Tuned TD(λ) outperforms QETR by {best_accuracy - qetr_acc:.1%}")
    
    results['best_td_lambda'] = best_lambda
    results['best_td_accuracy'] = best_accuracy
    results['scenario'] = scenario.name
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scenario_results(results: Dict, output_path: Path):
    """Plot TD(λ) comparison for a single scenario"""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract TD results
    lambda_values = []
    td_accuracies = []
    td_stds = []
    
    for key, val in results.items():
        if key.startswith('TD(λ='):
            lam = val['lambda']
            lambda_values.append(lam)
            td_accuracies.append(val['accuracy'])
            td_stds.append(val['std'])
    
    # Sort by lambda
    sorted_idx = np.argsort(lambda_values)
    lambda_values = [lambda_values[i] for i in sorted_idx]
    td_accuracies = [td_accuracies[i] for i in sorted_idx]
    td_stds = [td_stds[i] for i in sorted_idx]
    
    # Plot TD(λ) curve
    ax.errorbar(lambda_values, td_accuracies, yerr=td_stds,
                fmt='o-', color='#A23B72', label='TD(λ) - tuned', 
                markersize=8, capsize=4)
    
    # Plot QETR horizontal line
    qetr_acc = results['QETR-P31']['accuracy']
    qetr_std = results['QETR-P31']['std']
    qetr_lambda = results['QETR-P31']['lambda']
    
    ax.axhline(y=qetr_acc, color='#2E86AB', linestyle='-', 
               linewidth=2, label=f'QETR-P31 (λ={qetr_lambda:.4f})')
    ax.axhspan(qetr_acc - qetr_std, qetr_acc + qetr_std, 
               alpha=0.2, color='#2E86AB')
    
    # Mark QETR's effective lambda on x-axis
    ax.axvline(x=qetr_lambda, color='#2E86AB', linestyle=':', alpha=0.5)
    
    # Chance line
    scenario_name = results.get('scenario', 'easy')
    if scenario_name in SCENARIOS:
        n_actions = SCENARIOS[scenario_name].n_actions
    else:
        n_actions = 4
    chance = 1.0 / n_actions
    ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.5, label=f'Chance (1/{n_actions})')
    
    ax.set_xlabel('λ (eligibility decay parameter)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'QETR vs TD(λ): {results.get("scenario", "unknown")}', fontsize=12)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.45, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path / f'td_comparison_{results["scenario"]}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}/td_comparison_{results['scenario']}.png")


def plot_all_scenarios(all_results: Dict[str, Dict], output_path: Path):
    """Plot comparison across all scenarios"""
    
    scenarios = list(all_results.keys())
    n_scenarios = len(scenarios)
    
    fig, axes = plt.subplots(2, (n_scenarios + 1) // 2, 
                             figsize=(5 * ((n_scenarios + 1) // 2), 8))
    axes = axes.flatten()
    
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Extract TD results
        lambda_values = []
        td_accuracies = []
        
        for key, val in results.items():
            if key.startswith('TD(λ='):
                lambda_values.append(val['lambda'])
                td_accuracies.append(val['accuracy'])
        
        # Sort
        sorted_idx = np.argsort(lambda_values)
        lambda_values = [lambda_values[i] for i in sorted_idx]
        td_accuracies = [td_accuracies[i] for i in sorted_idx]
        
        # Plot
        ax.plot(lambda_values, td_accuracies, 'o-', color='#A23B72', 
                markersize=6, label='TD(λ)')
        
        qetr_acc = results['QETR-P31']['accuracy']
        ax.axhline(y=qetr_acc, color='#2E86AB', linewidth=2, label='QETR')
        
        ax.set_title(scenario_name, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.45, 1.0)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused axes
    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('QETR vs TD(λ) Across Scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'all_scenarios_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}/all_scenarios_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QETR Scenario Experiments')
    parser.add_argument('--scenario', type=str, default='easy',
                       help='Scenario to run (or "all")')
    parser.add_argument('--list', action='store_true',
                       help='List available scenarios')
    parser.add_argument('--all', action='store_true',
                       help='Run all scenarios')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # List scenarios
    if args.list:
        print("\nAvailable scenarios:")
        print("-" * 60)
        for name, scenario in SCENARIOS.items():
            print(f"  {name:20s} - {scenario.description}")
        return
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("qetr_framework/experiments/results") / f"qetr_scenarios_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("QETR SCENARIO EXPERIMENTS")
    print("Testing: Does physics-derived λ match tuned λ?")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    
    # Run scenarios
    if args.all:
        scenarios_to_run = list(SCENARIOS.keys())
    else:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(SCENARIOS.keys())}")
            return
        scenarios_to_run = [args.scenario]
    
    all_results = {}
    
    for scenario_name in scenarios_to_run:
        scenario = SCENARIOS[scenario_name]
        results = run_td_comparison(scenario)
        all_results[scenario_name] = results
        
        # Plot individual scenario
        plot_scenario_results(results, output_path)
    
    # Plot all scenarios together if multiple
    if len(all_results) > 1:
        plot_all_scenarios(all_results, output_path)
    
    # Save results
    # Convert numpy arrays for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj
    
    results_json = convert_for_json(all_results)
    results_json['timestamp'] = timestamp
    results_json['physics'] = {
        'T2_P31': PHYSICS.T2_P31,
        'T2_P32': PHYSICS.T2_P32,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {output_path}/results.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for scenario_name, results in all_results.items():
        qetr = results['QETR-P31']['accuracy']
        best_td = results['best_td_accuracy']
        best_lam = results['best_td_lambda']
        
        status = "✓" if qetr >= best_td - 0.02 else "✗"
        print(f"{status} {scenario_name:20s}: QETR={qetr:.1%}, Best TD(λ={best_lam})={best_td:.1%}")


if __name__ == "__main__":
    main()