"""
Coordination Environment
=========================

Multi-agent coordination task where agents must learn correlated responses
from only global scalar reward.

The challenge: N agents must find a joint configuration that matches
a hidden target pattern. Only scalar global reward is provided -
no per-agent error signals (no backprop).

This is provably hard for independent learners:
- 2^N possible configurations
- Random search takes exponential time
- Gradient-based methods need per-agent gradients (not available)

The quantum mechanism provides a solution:
- Co-active agents become entangled
- Shared quantum state means coordinated collapse on reward
- This is coordination through physics, not algorithm

Success criteria:
- QETR with entanglement: O(N) or O(N²) to learn
- Independent learners: O(2^N) (exponential)
- Centralized critic: O(N) but needs per-agent errors (biologically implausible)

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set


@dataclass
class CoordinationConfig:
    """
    Configuration for coordination task
    """
    # Number of agents (synapses)
    n_agents: int = 10
    
    # Target pattern (None = random)
    target_pattern: np.ndarray = None
    
    # Reward structure
    match_reward: float = 1.0      # Per matching agent
    full_match_bonus: float = 5.0  # Extra for all matching
    mismatch_penalty: float = 0.0  # Penalty for mismatches
    
    # Reward delivery
    immediate_reward: bool = False  # If True, reward after each action
    delayed_reward: bool = True     # If True, reward after delay
    delay_s: float = 30.0           # Delay in seconds
    
    # Timing
    dt: float = 0.1
    episode_length: float = 300.0
    
    # Difficulty
    noise_level: float = 0.0  # Action execution noise
    target_changes: bool = False  # Whether target changes during episode


@dataclass
class CoordinationState:
    """
    Environment state
    """
    time: float = 0.0
    target: np.ndarray = None
    last_actions: np.ndarray = None
    
    # Pending rewards
    pending_rewards: List[Tuple[float, float, np.ndarray]] = field(default_factory=list)
    
    # Statistics
    n_rounds: int = 0
    n_perfect_matches: int = 0
    cumulative_reward: float = 0.0
    match_history: List[float] = field(default_factory=list)


class CoordinationEnvironment:
    """
    Multi-agent coordination environment.
    
    N agents choose binary actions. Reward depends on how many agents
    match a hidden target pattern. Only global scalar reward provided.
    
    Key insight: This task exposes the coordination problem.
    Without explicit per-agent error signals, how do agents learn
    to act together coherently?
    
    The quantum answer: Entanglement provides implicit coordination.
    Agents that act together share quantum state. When reward arrives,
    all correlated agents update consistently.
    
    Usage:
        env = CoordinationEnvironment(config)
        reward, info = env.submit_actions(actions)
    """
    
    def __init__(self, config: Optional[CoordinationConfig] = None, seed: Optional[int] = None):
        """
        Initialize environment
        
        Args:
            config: Task configuration
            seed: Random seed
        """
        self.config = config or CoordinationConfig()
        self.rng = np.random.default_rng(seed)
        
        # Initialize target pattern
        if self.config.target_pattern is not None:
            target = self.config.target_pattern
        else:
            target = self.rng.integers(0, 2, size=self.config.n_agents)
        
        self.state = CoordinationState(
            target=target,
            last_actions=np.zeros(self.config.n_agents, dtype=int),
        )
    
    @property
    def n_agents(self) -> int:
        return self.config.n_agents
    
    @property
    def target(self) -> np.ndarray:
        return self.state.target
    
    def reset(self) -> Dict:
        """Reset environment"""
        if self.config.target_pattern is not None:
            target = self.config.target_pattern
        else:
            target = self.rng.integers(0, 2, size=self.config.n_agents)
        
        self.state = CoordinationState(
            target=target,
            last_actions=np.zeros(self.config.n_agents, dtype=int),
        )
        
        return {'n_agents': self.n_agents}
    
    def submit_actions(self, actions: np.ndarray) -> Tuple[float, Dict]:
        """
        Submit joint actions from all agents
        
        Args:
            actions: Binary array of length n_agents
            
        Returns:
            reward: Scalar global reward
            info: Additional information
        """
        assert len(actions) == self.n_agents
        
        # Apply noise if configured
        if self.config.noise_level > 0:
            flip_mask = self.rng.random(self.n_agents) < self.config.noise_level
            actions = actions.copy()
            actions[flip_mask] = 1 - actions[flip_mask]
        
        # Store actions
        self.state.last_actions = actions.copy()
        self.state.n_rounds += 1
        
        # Calculate match score
        matches = np.sum(actions == self.state.target)
        match_fraction = matches / self.n_agents
        is_perfect = (matches == self.n_agents)
        
        if is_perfect:
            self.state.n_perfect_matches += 1
        
        # Calculate reward
        reward_value = self._calculate_reward(matches, is_perfect)
        
        # Handle reward delivery
        if self.config.immediate_reward:
            reward = reward_value
            self.state.cumulative_reward += reward
        elif self.config.delayed_reward:
            # Schedule for later
            delivery_time = self.state.time + self.config.delay_s
            self.state.pending_rewards.append((delivery_time, reward_value, actions.copy()))
            reward = 0.0
        else:
            reward = 0.0
        
        self.state.match_history.append(match_fraction)
        
        info = {
            'matches': matches,
            'match_fraction': match_fraction,
            'is_perfect': is_perfect,
            'reward_value': reward_value,
            'time': self.state.time,
        }
        
        return reward, info
    
    def step_time(self, dt: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Advance time, potentially deliver delayed rewards
        
        Args:
            dt: Time step (uses config default if None)
            
        Returns:
            reward: Any delivered rewards
            info: State information
        """
        if dt is None:
            dt = self.config.dt
        
        self.state.time += dt
        reward = self._deliver_due_rewards()
        
        # Maybe change target
        if self.config.target_changes:
            self._maybe_change_target()
        
        info = {
            'time': self.state.time,
            'pending_rewards': len(self.state.pending_rewards),
        }
        
        return reward, info
    
    def _calculate_reward(self, matches: int, is_perfect: bool) -> float:
        """Calculate reward from match count"""
        cfg = self.config
        
        # Base reward from matches
        reward = matches * cfg.match_reward
        
        # Bonus for perfect match
        if is_perfect:
            reward += cfg.full_match_bonus
        
        # Penalty for mismatches
        mismatches = self.n_agents - matches
        reward -= mismatches * cfg.mismatch_penalty
        
        return reward
    
    def _deliver_due_rewards(self) -> float:
        """Deliver any pending rewards"""
        reward = 0.0
        delivered = []
        
        for (delivery_time, value, actions) in self.state.pending_rewards:
            if self.state.time >= delivery_time:
                reward += value
                delivered.append((delivery_time, value, actions))
                self.state.cumulative_reward += value
        
        for d in delivered:
            self.state.pending_rewards.remove(d)
        
        return reward
    
    def _maybe_change_target(self):
        """Potentially change target pattern"""
        # Change with small probability each step
        if self.rng.random() < 0.001:  # ~0.1% per step
            self.state.target = self.rng.integers(0, 2, size=self.n_agents)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'n_rounds': self.state.n_rounds,
            'n_perfect': self.state.n_perfect_matches,
            'perfect_rate': self.state.n_perfect_matches / max(1, self.state.n_rounds),
            'cumulative_reward': self.state.cumulative_reward,
            'mean_match_fraction': np.mean(self.state.match_history) if self.state.match_history else 0.0,
            'last_match_fraction': self.state.match_history[-1] if self.state.match_history else 0.0,
            'pending_rewards': len(self.state.pending_rewards),
            'time': self.state.time,
        }
    
    def get_target(self) -> np.ndarray:
        """
        Get target pattern (for evaluation - not available to agents!)
        """
        return self.state.target.copy()


# =============================================================================
# DIFFICULTY PRESETS
# =============================================================================

def make_easy_coordination(n_agents: int = 5, seed: Optional[int] = None) -> CoordinationEnvironment:
    """
    Easy coordination task
    - Few agents (5)
    - Immediate reward
    """
    config = CoordinationConfig(
        n_agents=n_agents,
        immediate_reward=True,
        delayed_reward=False,
        match_reward=0.2,
        full_match_bonus=1.0,
    )
    return CoordinationEnvironment(config, seed)


def make_medium_coordination(n_agents: int = 10, seed: Optional[int] = None) -> CoordinationEnvironment:
    """
    Medium coordination task
    - 10 agents
    - Short delay (10s)
    """
    config = CoordinationConfig(
        n_agents=n_agents,
        immediate_reward=False,
        delayed_reward=True,
        delay_s=10.0,
        match_reward=0.1,
        full_match_bonus=2.0,
    )
    return CoordinationEnvironment(config, seed)


def make_hard_coordination(n_agents: int = 10, seed: Optional[int] = None) -> CoordinationEnvironment:
    """
    Hard coordination task
    - 10 agents
    - Long delay (30s) - requires quantum coherence
    """
    config = CoordinationConfig(
        n_agents=n_agents,
        immediate_reward=False,
        delayed_reward=True,
        delay_s=30.0,
        match_reward=0.1,
        full_match_bonus=5.0,
    )
    return CoordinationEnvironment(config, seed)


def make_scaling_test(n_agents: int, seed: Optional[int] = None) -> CoordinationEnvironment:
    """
    Scaling test - vary n_agents to test O(N) vs O(2^N) scaling
    """
    config = CoordinationConfig(
        n_agents=n_agents,
        immediate_reward=False,
        delayed_reward=True,
        delay_s=30.0,
        match_reward=0.1,
        full_match_bonus=float(n_agents),
    )
    return CoordinationEnvironment(config, seed)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COORDINATION ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test 1: Basic functionality
    print("\n--- Test 1: Basic Coordination ---")
    env = make_easy_coordination(n_agents=5, seed=42)
    env.reset()
    
    print(f"Target pattern: {env.target}")
    
    # Try some random actions
    for trial in range(5):
        actions = np.random.randint(0, 2, size=5)
        reward, info = env.submit_actions(actions)
        print(f"Trial {trial+1}: actions={actions}, matches={info['matches']}, reward={reward:.2f}")
    
    # Test 2: Perfect match
    print("\n--- Test 2: Perfect Match ---")
    env.reset()
    print(f"Target: {env.target}")
    reward, info = env.submit_actions(env.target.copy())
    print(f"Perfect match reward: {reward:.2f} (perfect={info['is_perfect']})")
    
    # Test 3: Delayed reward
    print("\n--- Test 3: Delayed Reward ---")
    env = make_hard_coordination(n_agents=10, seed=42)
    env.reset()
    
    print(f"Target: {env.target}")
    
    # Submit perfect match
    reward, info = env.submit_actions(env.target.copy())
    print(f"Immediate reward: {reward} (should be 0, delayed)")
    
    # Wait for reward
    for t in range(int(30 / 0.1)):  # 30 seconds
        reward, _ = env.step_time()
        if reward > 0:
            print(f"Received delayed reward {reward:.2f} at t={env.state.time:.1f}s")
            break
    
    # Test 4: Scaling analysis
    print("\n--- Test 4: Search Space Scaling ---")
    for n in [5, 10, 15, 20]:
        search_space = 2 ** n
        print(f"N={n} agents: {search_space:,} possible configurations")
    
    print("\nRandom search would need O(2^N) trials.")
    print("QETR with entanglement should need O(N) or O(N²).")
    
    stats = env.get_statistics()
    print(f"\nFinal stats: {stats}")
    
    print("\n--- Done ---")