"""
Quantum Eligibility Trace Reinforcement Learning (QETR) Agent
==============================================================

The main RL agent using physics-derived eligibility traces.

Key difference from standard TD(λ):
- λ is not a hyperparameter - it's derived from T2
- T2 comes from nuclear spin physics
- Isotope selection is the only "tunable" parameter (and it's discrete)

This agent demonstrates:
1. No hyperparameter tuning needed for eligibility decay
2. Isotope substitution (P32) breaks learning predictably
3. Performance matches optimally-tuned TD(λ) without tuning

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from abc import ABC, abstractmethod


# To import from quantum_eligibility (sibling folder within entanglement)
from ..quantum_eligibility.physics import QuantumEligibilityPhysics, Isotope
from ..quantum_eligibility.synapse import QuantumSynapse, SynapseParameters

# To import from synapse folder (up two levels, then into synapse)
from ..synapse.network import QuantumSynapseNetwork, EntanglementParameters


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for RL agents
    """
    
    @abstractmethod
    def select_action(self, observation: Dict) -> int:
        """Select action given observation"""
        pass
    
    @abstractmethod
    def update(self, action: int, reward: float, dt: float):
        """Update agent after action and reward"""
        pass
    
    @abstractmethod
    def step_time(self, dt: float):
        """Advance time (for eligibility decay)"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent for new episode"""
        pass
    
    @abstractmethod
    def get_action_values(self) -> np.ndarray:
        """Get current value estimates for all actions"""
        pass


# =============================================================================
# QETR AGENT
# =============================================================================

@dataclass
class QETRConfig:
    """
    QETR Agent configuration
    
    Note: Most important parameters come from physics, not here.
    """
    # Learning
    learning_rate: float = 0.1
    
    # Exploration
    epsilon: float = 0.1  # For ε-greedy
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Temperature for softmax action selection
    temperature: float = 1.0
    
    # Action selection method
    action_selection: str = "epsilon_greedy"  # or "softmax"
    
    # Activation strength on action
    activation_strength: float = 1.0


class QETRAgent(BaseAgent):
    """
    Quantum Eligibility Trace Reinforcement Learning Agent
    
    Uses physics-derived eligibility traces for temporal credit assignment.
    
    Key mechanism:
    1. Each action has an associated quantum synapse
    2. Taking action activates that synapse (creates eligibility)
    3. Eligibility decays with physics-derived T2
    4. Reward updates all eligible synapses (three-factor learning)
    
    The T2 parameter is NOT tuned - it comes from nuclear spin physics.
    
    Usage:
        agent = QETRAgent(n_actions=4, isotope=Isotope.P31)
        
        for step in episode:
            action = agent.select_action(obs)
            reward, info = env.step(action)
            agent.update(action, reward, dt=0.1)
            agent.step_time(dt=0.1)
    """
    
    def __init__(self,
                 n_actions: int,
                 isotope: Isotope = Isotope.P31,
                 config: Optional[QETRConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize QETR agent
        
        Args:
            n_actions: Number of available actions
            isotope: Phosphorus isotope (determines T2)
            config: Agent configuration
            seed: Random seed
        """
        self.n_actions = n_actions
        self.isotope = isotope
        self.config = config or QETRConfig()
        
        self.rng = np.random.default_rng(seed)
        
        # Create quantum synapses (one per action)
        syn_params = SynapseParameters(learning_rate=self.config.learning_rate)
        seeds = self.rng.integers(0, 2**31, size=n_actions)
        
        self.synapses = [
            QuantumSynapse(isotope=isotope, params=syn_params, seed=int(s))
            for s in seeds
        ]
        
        # Physics (shared across synapses)
        self.physics = self.synapses[0].physics
        self.T2 = self.synapses[0].T2
        
        # Current exploration rate
        self.epsilon = self.config.epsilon
        
        # Statistics
        self.n_actions_taken = 0
        self.n_updates = 0
    
    def select_action(self, observation: Dict = None) -> int:
        """
        Select action using ε-greedy or softmax
        
        Args:
            observation: Not used in basic agent (could add state-dependence)
            
        Returns:
            Selected action index
        """
        if self.config.action_selection == "epsilon_greedy":
            return self._select_epsilon_greedy()
        else:
            return self._select_softmax()
    
    def _select_epsilon_greedy(self) -> int:
        """ε-greedy action selection"""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            values = self.get_action_values()
            return int(np.argmax(values))
    
    def _select_softmax(self) -> int:
        """Softmax action selection"""
        values = self.get_action_values()
        
        # Temperature-scaled softmax
        exp_values = np.exp(values / self.config.temperature)
        probs = exp_values / np.sum(exp_values)
        
        return self.rng.choice(self.n_actions, p=probs)
    
    def update(self, action: int, reward: float, dt: float = 0.0):
        """
        Update agent after taking action and receiving reward
        
        This implements three-factor learning:
        1. Action → activate corresponding synapse
        2. Time passes → eligibility decays with T2
        3. Reward → update weights proportional to eligibility
        
        Args:
            action: Action that was taken
            reward: Reward received
            dt: Time since action (for immediate update)
        """
        # Activate synapse for chosen action
        self.synapses[action].activate(self.config.activation_strength)
        
        # If time passed, evolve all synapses
        if dt > 0:
            self.step_time(dt)
        
        # Apply reward to ALL synapses (three-factor rule)
        if reward != 0:
            for syn in self.synapses:
                syn.apply_reward(reward)
        
        self.n_actions_taken += 1
        
        # Decay exploration
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
    
    def step_time(self, dt: float):
        """
        Advance time for all synapses
        
        This decays eligibility traces with physics-derived T2.
        """
        for syn in self.synapses:
            syn.step(dt)
    
    def reset(self):
        """Reset for new episode (keep learned weights)"""
        for syn in self.synapses:
            syn.reset()
        self.epsilon = self.config.epsilon
    
    def reset_weights(self):
        """Full reset including weights"""
        for syn in self.synapses:
            syn.reset_weight()
        self.epsilon = self.config.epsilon
        self.n_actions_taken = 0
    
    def get_action_values(self) -> np.ndarray:
        """Get current Q-values (synapse weights)"""
        return np.array([syn.weight for syn in self.synapses])
    
    def get_eligibilities(self) -> np.ndarray:
        """Get current eligibility values"""
        return np.array([syn.eligibility for syn in self.synapses])
    
    def get_state(self) -> Dict:
        """Get agent state for analysis"""
        return {
            'isotope': self.isotope.value,
            'T2': self.T2,
            'n_actions': self.n_actions,
            'weights': self.get_action_values(),
            'eligibilities': self.get_eligibilities(),
            'epsilon': self.epsilon,
            'n_actions_taken': self.n_actions_taken,
        }
    
    def __repr__(self) -> str:
        return f"QETRAgent(n_actions={self.n_actions}, isotope={self.isotope.value}, T2={self.T2:.1f}s)"


# =============================================================================
# QETR NETWORK AGENT (for coordination tasks)
# =============================================================================

class QETRNetworkAgent:
    """
    QETR Agent with entanglement for coordination tasks
    
    Uses a network of quantum synapses where co-activated synapses
    become entangled, enabling coordinated learning.
    
    For coordination tasks: each "action dimension" is one synapse.
    """
    
    def __init__(self,
                 n_agents: int,
                 isotope: Isotope = Isotope.P31,
                 entanglement_params: Optional[EntanglementParameters] = None,
                 learning_rate: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize network agent
        
        Args:
            n_agents: Number of coordinating agents (synapses)
            isotope: Isotope for coherence
            entanglement_params: Inter-synapse coupling
            learning_rate: Learning rate
            seed: Random seed
        """
        self.n_agents = n_agents
        self.isotope = isotope
        
        syn_params = SynapseParameters(learning_rate=learning_rate)
        
        self.network = QuantumSynapseNetwork(
            n_synapses=n_agents,
            isotope=isotope,
            entanglement_params=entanglement_params,
            synapse_params=syn_params,
            seed=seed,
        )
        
        self.T2 = self.network.T2
        self.rng = np.random.default_rng(seed)
    
    def select_actions(self, temperature: float = 1.0) -> np.ndarray:
        """
        Select binary actions for all agents
        
        Uses sigmoid of weights as probability of action=1
        
        Args:
            temperature: Softmax temperature
            
        Returns:
            Binary action array
        """
        weights = self.network.get_weights()
        probs = 1.0 / (1.0 + np.exp(-weights / temperature))
        
        actions = (self.rng.random(self.n_agents) < probs).astype(int)
        
        return actions
    
    def update(self, actions: np.ndarray, reward: float):
        """
        Update network after joint action and reward
        
        Args:
            actions: Binary actions taken
            reward: Global scalar reward
        """
        # Activate synapses that took action=1
        active_indices = np.where(actions == 1)[0].tolist()
        if active_indices:
            self.network.activate(active_indices, strength=1.0)
        
        # Apply reward
        self.network.apply_reward(reward)
    
    def step_time(self, dt: float):
        """Advance time"""
        self.network.step(dt)
    
    def reset(self):
        """Reset for new episode"""
        self.network.reset()
    
    def get_weights(self) -> np.ndarray:
        """Get weights"""
        return self.network.get_weights()
    
    def get_entanglement(self) -> np.ndarray:
        """Get entanglement matrix"""
        return self.network.get_entanglement_graph()


# =============================================================================
# BASELINE: TD(λ) AGENT
# =============================================================================

class TDLambdaAgent(BaseAgent):
    """
    Standard TD(λ) agent for comparison
    
    This is the classical RL baseline. The key difference:
    λ is a hyperparameter that must be tuned for each task.
    
    Compare to QETR where the effective λ is derived from physics.
    """
    
    def __init__(self,
                 n_actions: int,
                 lambda_: float = 0.9,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 dt: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize TD(λ) agent
        
        Args:
            n_actions: Number of actions
            lambda_: Eligibility trace decay (HYPERPARAMETER - must tune!)
            alpha: Learning rate
            epsilon: Exploration rate
            dt: Timestep for trace decay
            seed: Random seed
        """
        self.n_actions = n_actions
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epsilon = epsilon
        self.dt = dt
        
        self.rng = np.random.default_rng(seed)
        
        # Q-values (weights)
        self.Q = np.zeros(n_actions)
        
        # Eligibility traces
        self.e = np.zeros(n_actions)
    
    def select_action(self, observation: Dict = None) -> int:
        """ε-greedy selection"""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return int(np.argmax(self.Q))
    
    def update(self, action: int, reward: float, dt: float = None):
        """
        Update Q-values using TD(λ)
        
        The key step: e ← λ × e
        This λ is a hyperparameter that must be tuned!
        """
        if dt is None:
            dt = self.dt
        
        # Update eligibility for chosen action
        self.e[action] = 1.0  # Replacing traces
        
        # TD update
        delta = reward  # Simplified: no next state value
        self.Q += self.alpha * delta * self.e
        
        # Decay traces (THIS IS THE HYPERPARAMETER)
        self.e *= self.lambda_
    
    def step_time(self, dt: float):
        """Decay traces over time"""
        # In discrete TD(λ), decay happens per step
        # For continuous time, decay = λ^(dt/dt_base)
        decay = self.lambda_ ** (dt / self.dt)
        self.e *= decay
    
    def reset(self):
        """Reset traces for new episode"""
        self.e = np.zeros(self.n_actions)
    
    def reset_weights(self):
        """Full reset"""
        self.Q = np.zeros(self.n_actions)
        self.e = np.zeros(self.n_actions)
    
    def get_action_values(self) -> np.ndarray:
        return self.Q.copy()
    
    def get_eligibilities(self) -> np.ndarray:
        return self.e.copy()
    
    def __repr__(self) -> str:
        return f"TDLambdaAgent(n_actions={self.n_actions}, λ={self.lambda_})"


# =============================================================================
# BASELINE: INDEPENDENT Q-LEARNER (for coordination)
# =============================================================================

class IndependentQLearner:
    """
    Independent Q-learners for coordination comparison
    
    Each agent learns independently using the same global reward.
    This is known to fail on coordination tasks because agents
    cannot attribute global reward to their individual actions.
    """
    
    def __init__(self,
                 n_agents: int,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize independent learners
        
        Args:
            n_agents: Number of independent learners
            alpha: Learning rate
            epsilon: Exploration rate
            seed: Random seed
        """
        self.n_agents = n_agents
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(seed)
        
        # Each agent has Q-value for action 0 vs 1
        self.Q = np.zeros((n_agents, 2))
    
    def select_actions(self) -> np.ndarray:
        """Select binary actions independently"""
        actions = np.zeros(self.n_agents, dtype=int)
        
        for i in range(self.n_agents):
            if self.rng.random() < self.epsilon:
                actions[i] = self.rng.integers(0, 2)
            else:
                actions[i] = np.argmax(self.Q[i])
        
        return actions
    
    def update(self, actions: np.ndarray, reward: float):
        """
        Update each agent with global reward
        
        The problem: each agent gets the same reward regardless
        of whether their individual action was correct.
        """
        for i, a in enumerate(actions):
            self.Q[i, a] += self.alpha * (reward - self.Q[i, a])
    
    def step_time(self, dt: float):
        """No time dynamics for basic Q-learning"""
        pass
    
    def reset(self):
        """Reset for new episode"""
        pass
    
    def reset_weights(self):
        """Full reset"""
        self.Q = np.zeros((self.n_agents, 2))
    
    def get_weights(self) -> np.ndarray:
        """Return preference for action=1"""
        return self.Q[:, 1] - self.Q[:, 0]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QETR AGENT TEST")
    print("=" * 60)
    
    # Test 1: Basic QETR
    print("\n--- Test 1: Basic QETR Agent ---")
    agent = QETRAgent(n_actions=4, isotope=Isotope.P31, seed=42)
    print(agent)
    print(f"Physics-derived T2: {agent.T2:.1f}s")
    
    # Simulate learning
    for trial in range(10):
        action = agent.select_action()
        reward = 1.0 if action == 2 else 0.0  # Action 2 is correct
        agent.update(action, reward, dt=1.0)
        agent.step_time(dt=1.0)
    
    print(f"Learned Q-values: {agent.get_action_values()}")
    print(f"Best action: {np.argmax(agent.get_action_values())}")
    
    # Test 2: Isotope comparison
    print("\n--- Test 2: Isotope Comparison ---")
    
    for isotope in [Isotope.P31, Isotope.P32]:
        agent = QETRAgent(n_actions=4, isotope=isotope, seed=42)
        
        # Learn with 30s delay between action and reward
        agent.select_action()  # Dummy action
        agent.synapses[2].activate(1.0)  # Activate correct action
        
        # Wait 30 seconds
        for _ in range(300):
            agent.step_time(dt=0.1)
        
        # Apply reward
        eligibility_before = agent.synapses[2].eligibility
        agent.synapses[2].apply_reward(reward=1.0)
        weight_change = agent.synapses[2].weight
        
        print(f"{isotope.value}: T2={agent.T2:.1f}s, eligibility@30s={eligibility_before:.4f}, Δw={weight_change:.4f}")
    
    # Test 3: TD(λ) baseline
    print("\n--- Test 3: TD(λ) Baseline ---")
    td_agent = TDLambdaAgent(n_actions=4, lambda_=0.95, seed=42)
    print(td_agent)
    
    # Same learning task
    for trial in range(10):
        action = td_agent.select_action()
        reward = 1.0 if action == 2 else 0.0
        td_agent.update(action, reward)
    
    print(f"Learned Q-values: {td_agent.get_action_values()}")
    
    # Test 4: Network agent for coordination
    print("\n--- Test 4: Network Agent ---")
    net_agent = QETRNetworkAgent(n_agents=5, isotope=Isotope.P31, seed=42)
    print(f"Network with {net_agent.n_agents} agents, T2={net_agent.T2:.1f}s")
    
    # Simulate coordination
    target = np.array([1, 0, 1, 1, 0])
    
    for trial in range(20):
        actions = net_agent.select_actions()
        matches = np.sum(actions == target)
        reward = matches / len(target)
        net_agent.update(actions, reward)
        net_agent.step_time(dt=1.0)
    
    print(f"Final weights: {net_agent.get_weights()}")
    print(f"Target pattern: {target}")
    final_actions = (net_agent.get_weights() > 0).astype(int)
    print(f"Learned pattern: {final_actions}")
    
    print("\n--- Done ---")