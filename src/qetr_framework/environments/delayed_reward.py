"""
Delayed Reward Environment
===========================

Environment for testing temporal credit assignment.

The challenge: choose the correct action, but reward arrives with
variable delay (30-90 seconds). The agent must remember which action
led to which outcome over this delay.

This specifically targets the 60-100 second window where:
- Classical biochemical mechanisms (CaMKII) fail (τ ~ 5-15s)
- Quantum coherence (P31 dimers) succeeds (T2 ~ 25-67s)

Success criteria:
- P31-based agent should learn reliably
- P32-based agent should fail (eligibility decays before reward)
- Standard TD(λ) requires task-specific tuning of λ

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum


@dataclass
class DelayedRewardConfig:
    """
    Configuration for delayed reward task
    """
    # Action space
    n_actions: int = 4
    
    # Delay parameters (seconds)
    delay_min: float = 30.0
    delay_max: float = 90.0
    delay_distribution: str = "uniform"  # or "gaussian"
    
    # Reward structure
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0  # Could be negative for punishment
    
    # Task dynamics
    correct_action_changes: bool = False  # Whether target changes
    change_interval: float = 1000.0       # Seconds between changes
    
    # Timing
    dt: float = 0.1  # Time step in seconds
    episode_length: float = 300.0  # 5 minutes per episode
    
    # Observation
    provide_time_signal: bool = False  # Whether to give time since action


@dataclass 
class EnvironmentState:
    """
    Internal state of the environment
    """
    time: float = 0.0
    correct_action: int = 0
    
    # Pending rewards: list of (delivery_time, reward_value, action_taken)
    pending_rewards: List[Tuple[float, float, int]] = field(default_factory=list)
    
    # Statistics
    n_actions_taken: int = 0
    n_correct_actions: int = 0
    n_rewards_delivered: int = 0
    cumulative_reward: float = 0.0
    
    # Per-action statistics
    action_counts: np.ndarray = None
    action_rewards: np.ndarray = None


class DelayedRewardEnvironment:
    """
    Environment with delayed reward delivery.
    
    Task: Agent chooses one of N actions. One action is "correct" and
    triggers a reward, but the reward arrives after a variable delay.
    
    The delay is the key challenge:
    - Short delay (< 5s): Easy, classical mechanisms work
    - Medium delay (10-30s): Challenging, requires good eligibility traces
    - Long delay (30-90s): Hard, only quantum coherence can bridge this gap
    
    This task is designed to demonstrate:
    1. Physics-derived T2 provides appropriate eligibility window
    2. Isotope substitution (P32) breaks learning
    3. TD(λ) requires task-specific λ tuning, QETR doesn't
    
    Usage:
        env = DelayedRewardEnvironment()
        action = agent.select_action(obs)
        reward, info = env.step(action)
    """
    
    def __init__(self, config: Optional[DelayedRewardConfig] = None, seed: Optional[int] = None):
        """
        Initialize environment
        
        Args:
            config: Task configuration
            seed: Random seed
        """
        self.config = config or DelayedRewardConfig()
        self.rng = np.random.default_rng(seed)
        
        # Initialize state
        self.state = EnvironmentState(
            correct_action=self.rng.integers(0, self.config.n_actions),
            action_counts=np.zeros(self.config.n_actions),
            action_rewards=np.zeros(self.config.n_actions),
        )
    
    @property
    def n_actions(self) -> int:
        return self.config.n_actions
    
    @property
    def time(self) -> float:
        return self.state.time
    
    def reset(self) -> Dict:
        """
        Reset environment to initial state
        
        Returns:
            Initial observation
        """
        self.state = EnvironmentState(
            correct_action=self.rng.integers(0, self.config.n_actions),
            action_counts=np.zeros(self.config.n_actions),
            action_rewards=np.zeros(self.config.n_actions),
        )
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[float, Dict]:
        """
        Take action, advance time, potentially receive delayed reward
        
        Args:
            action: Action to take (0 to n_actions-1)
            
        Returns:
            reward: Immediate reward (usually 0, delayed rewards arrive later)
            info: Dictionary with additional information
        """
        assert 0 <= action < self.config.n_actions, f"Invalid action {action}"
        
        # Time advances
        self.state.time += self.config.dt
        
        # Record action
        self.state.n_actions_taken += 1
        self.state.action_counts[action] += 1
        
        # Check if action is correct
        is_correct = (action == self.state.correct_action)
        
        if is_correct:
            self.state.n_correct_actions += 1
            
            # Schedule delayed reward
            delay = self._sample_delay()
            delivery_time = self.state.time + delay
            self.state.pending_rewards.append(
                (delivery_time, self.config.correct_reward, action)
            )
        elif self.config.incorrect_reward != 0:
            # Optional: immediate feedback for incorrect actions
            # (usually we don't want this - pure delayed reward)
            pass
        
        # Check for reward delivery
        reward = self._deliver_due_rewards()
        
        # Maybe change correct action
        if self.config.correct_action_changes:
            self._maybe_change_target()
        
        # Check episode end
        done = self.state.time >= self.config.episode_length
        
        info = {
            'time': self.state.time,
            'is_correct': is_correct,
            'pending_rewards': len(self.state.pending_rewards),
            'correct_action': self.state.correct_action,
            'done': done,
            'accuracy': self.state.n_correct_actions / max(1, self.state.n_actions_taken),
        }
        
        return reward, info
    
    def step_no_action(self) -> Tuple[float, Dict]:
        """
        Advance time without taking action (for eligibility decay)
        
        Returns:
            reward: Any delayed rewards that arrived
            info: State information
        """
        self.state.time += self.config.dt
        reward = self._deliver_due_rewards()
        
        info = {
            'time': self.state.time,
            'pending_rewards': len(self.state.pending_rewards),
        }
        
        return reward, info
    
    def _sample_delay(self) -> float:
        """Sample reward delay from configured distribution"""
        cfg = self.config
        
        if cfg.delay_distribution == "uniform":
            return self.rng.uniform(cfg.delay_min, cfg.delay_max)
        
        elif cfg.delay_distribution == "gaussian":
            mean = (cfg.delay_min + cfg.delay_max) / 2
            std = (cfg.delay_max - cfg.delay_min) / 4  # 95% within range
            delay = self.rng.normal(mean, std)
            return np.clip(delay, cfg.delay_min, cfg.delay_max)
        
        else:
            raise ValueError(f"Unknown delay distribution: {cfg.delay_distribution}")
    
    def _deliver_due_rewards(self) -> float:
        """Check and deliver any rewards that are due"""
        reward = 0.0
        delivered = []
        
        for (delivery_time, value, action) in self.state.pending_rewards:
            if self.state.time >= delivery_time:
                reward += value
                delivered.append((delivery_time, value, action))
                
                # Update statistics
                self.state.n_rewards_delivered += 1
                self.state.cumulative_reward += value
                self.state.action_rewards[action] += value
        
        # Remove delivered rewards
        for d in delivered:
            self.state.pending_rewards.remove(d)
        
        return reward
    
    def _maybe_change_target(self):
        """Potentially change the correct action"""
        if self.state.time % self.config.change_interval < self.config.dt:
            old_correct = self.state.correct_action
            new_correct = self.rng.integers(0, self.config.n_actions)
            self.state.correct_action = new_correct
            
            # Clear pending rewards for old correct action
            # (In a more realistic task, might keep them)
            self.state.pending_rewards = [
                r for r in self.state.pending_rewards 
                if r[2] != old_correct
            ]
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        obs = {
            'n_actions': self.config.n_actions,
        }
        
        if self.config.provide_time_signal:
            obs['time'] = self.state.time
        
        return obs
    
    def get_statistics(self) -> Dict:
        """Get comprehensive task statistics"""
        return {
            'time': self.state.time,
            'n_actions': self.state.n_actions_taken,
            'n_correct': self.state.n_correct_actions,
            'n_rewards': self.state.n_rewards_delivered,
            'cumulative_reward': self.state.cumulative_reward,
            'accuracy': self.state.n_correct_actions / max(1, self.state.n_actions_taken),
            'action_distribution': self.state.action_counts.copy(),
            'action_rewards': self.state.action_rewards.copy(),
            'pending_count': len(self.state.pending_rewards),
        }


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def make_short_delay_task(seed: Optional[int] = None) -> DelayedRewardEnvironment:
    """
    Short delay task (5-15s) - classical mechanisms should work
    """
    config = DelayedRewardConfig(
        delay_min=5.0,
        delay_max=15.0,
        episode_length=120.0,
    )
    return DelayedRewardEnvironment(config, seed)


def make_medium_delay_task(seed: Optional[int] = None) -> DelayedRewardEnvironment:
    """
    Medium delay task (15-45s) - challenges classical mechanisms
    """
    config = DelayedRewardConfig(
        delay_min=15.0,
        delay_max=45.0,
        episode_length=180.0,
    )
    return DelayedRewardEnvironment(config, seed)


def make_long_delay_task(seed: Optional[int] = None) -> DelayedRewardEnvironment:
    """
    Long delay task (30-90s) - requires quantum coherence
    
    This is the key benchmark: delays target the 60-100s window
    that exceeds classical biochemical mechanisms.
    """
    config = DelayedRewardConfig(
        delay_min=30.0,
        delay_max=90.0,
        episode_length=300.0,
    )
    return DelayedRewardEnvironment(config, seed)


def make_extreme_delay_task(seed: Optional[int] = None) -> DelayedRewardEnvironment:
    """
    Extreme delay task (60-120s) - at the edge of quantum coherence
    
    Even P31 dimers will struggle here (T2 ~ 25-67s).
    Tests the limits of the quantum mechanism.
    """
    config = DelayedRewardConfig(
        delay_min=60.0,
        delay_max=120.0,
        episode_length=600.0,
    )
    return DelayedRewardEnvironment(config, seed)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DELAYED REWARD ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test basic functionality
    print("\n--- Test 1: Basic Operation ---")
    env = make_long_delay_task(seed=42)
    obs = env.reset()
    print(f"Initialized with {env.n_actions} actions")
    print(f"Correct action: {env.state.correct_action}")
    
    # Take some actions
    total_reward = 0.0
    n_steps = 1000  # 100 seconds at dt=0.1
    
    for step in range(n_steps):
        # Random action every 10 steps
        if step % 100 == 0:  # Every 10 seconds
            action = np.random.randint(0, env.n_actions)
            reward, info = env.step(action)
        else:
            reward, info = env.step_no_action()
        
        total_reward += reward
        
        if reward > 0:
            print(f"  Step {step} (t={env.time:.1f}s): Received reward {reward}")
    
    stats = env.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Actions taken: {stats['n_actions']}")
    print(f"  Correct actions: {stats['n_correct']}")
    print(f"  Rewards delivered: {stats['n_rewards']}")
    print(f"  Total reward: {stats['cumulative_reward']:.1f}")
    print(f"  Pending rewards: {stats['pending_count']}")
    
    # Test 2: Different delay configurations
    print("\n--- Test 2: Delay Configurations ---")
    
    configs = [
        ("Short (5-15s)", make_short_delay_task),
        ("Medium (15-45s)", make_medium_delay_task),
        ("Long (30-90s)", make_long_delay_task),
        ("Extreme (60-120s)", make_extreme_delay_task),
    ]
    
    for name, make_fn in configs:
        env = make_fn(seed=42)
        
        # Sample some delays
        delays = [env._sample_delay() for _ in range(100)]
        
        print(f"{name}: mean delay = {np.mean(delays):.1f}s, std = {np.std(delays):.1f}s")
    
    print("\n--- Done ---")