"""
Quantum Synapse Module
======================

Implements a single synapse with quantum eligibility trace.

The synapse has three key state variables:
1. weight: The synaptic strength (learnable)
2. eligibility: The trace value (decays with T2)  
3. singlet_probability: Quantum coherence level (0.25=thermal, 1.0=pure singlet)

The eligibility trace is the molecular implementation of the "tag" in
three-factor learning rules. Here, it's implemented by nuclear spin
coherence in calcium phosphate dimers.

Key mechanism:
- Synaptic activation → dimer formation → eligibility created
- Eligibility decays with time constant T2 (isotope-dependent)
- Reward signal (dopamine) → converts eligibility to weight change
- This is "measurement" that collapses quantum exploration

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum

from .physics import QuantumEligibilityPhysics, Isotope


@dataclass
class SynapseState:
    """
    Complete state of a quantum synapse
    
    Tracks both classical (weight) and quantum (coherence) variables
    """
    # Classical state
    weight: float = 0.0
    
    # Quantum eligibility trace
    eligibility: float = 0.0
    singlet_probability: float = 0.25  # Thermal equilibrium
    
    # Timing
    time_since_activation: float = np.inf
    
    # Statistics
    n_activations: int = 0
    n_weight_updates: int = 0
    cumulative_weight_change: float = 0.0


@dataclass
class SynapseParameters:
    """
    Parameters for synapse behavior
    
    Note: T2 is NOT here - it comes from physics
    """
    # Weight bounds
    weight_min: float = -10.0
    weight_max: float = 10.0
    weight_init: float = 0.0
    
    # Eligibility thresholds
    eligibility_threshold: float = 0.1  # Minimum for plasticity
    eligibility_saturation: float = 1.0  # Maximum eligibility
    
    # Learning rates
    learning_rate: float = 0.1
    
    # Singlet dynamics
    singlet_creation_rate: float = 0.75  # How fast singlet forms on activation
    singlet_thermal: float = 0.25        # Equilibrium value


class QuantumSynapse:
    """
    Single synapse with quantum eligibility trace
    
    This is the core computational unit of the QETR framework.
    
    The synapse implements three-factor learning:
    1. Presynaptic activity → creates eligibility trace
    2. Time passes → eligibility decays with T2
    3. Reward arrives → eligibility converted to weight change
    
    The key insight is that T2 (and thus the eligibility window) is
    determined by physics, not tuned as a hyperparameter.
    
    Usage:
        synapse = QuantumSynapse(isotope=Isotope.P31)
        synapse.activate(strength=1.0)   # Create eligibility
        synapse.step(dt=0.1)             # Time evolution
        synapse.apply_reward(reward=1.0) # Convert to weight change
    """
    
    def __init__(self, 
                 isotope: Isotope = Isotope.P31,
                 physics: Optional[QuantumEligibilityPhysics] = None,
                 params: Optional[SynapseParameters] = None,
                 seed: Optional[int] = None):
        """
        Initialize quantum synapse
        
        Args:
            isotope: Phosphorus isotope (determines T2)
            physics: Pre-configured physics, or None to create from isotope
            params: Synapse parameters, or None for defaults
            seed: Random seed for reproducibility
        """
        # Physics (determines T2)
        if physics is not None:
            self.physics = physics
        else:
            self.physics = QuantumEligibilityPhysics(isotope=isotope)
        
        # Parameters
        self.params = params or SynapseParameters()
        
        # State
        self.state = SynapseState(weight=self.params.weight_init)
        
        # Derived quantities (cached for efficiency)
        self._T2 = self.physics.get_T2_effective()
        self._decay_rate = self.physics.get_decay_rate()
        
        # Random state
        self.rng = np.random.default_rng(seed)
        
        # History (optional, for analysis)
        self._history_enabled = False
        self._history: Dict[str, List] = {}
    
    @property
    def T2(self) -> float:
        """Effective coherence time (seconds)"""
        return self._T2
    
    @property
    def weight(self) -> float:
        """Current synaptic weight"""
        return self.state.weight
    
    @property
    def eligibility(self) -> float:
        """Current eligibility trace value"""
        return self.state.eligibility
    
    @property
    def singlet_prob(self) -> float:
        """Current singlet probability (quantum coherence)"""
        return self.state.singlet_probability
    
    @property
    def is_eligible(self) -> bool:
        """Whether synapse is currently eligible for plasticity"""
        return self.state.eligibility > self.params.eligibility_threshold
    
    @property
    def is_entangled(self) -> bool:
        """Whether quantum entanglement is preserved (P_singlet > 0.5)"""
        return self.state.singlet_probability > self.physics.get_entanglement_threshold()
    
    def reset(self):
        """Reset synapse to initial state (keeps weight)"""
        self.state.eligibility = 0.0
        self.state.singlet_probability = self.params.singlet_thermal
        self.state.time_since_activation = np.inf
    
    def reset_weight(self):
        """Full reset including weight"""
        self.state = SynapseState(weight=self.params.weight_init)
    
    def set_isotope(self, isotope: Isotope):
        """
        Change isotope (for ablation studies)
        
        This recalculates T2 from physics.
        """
        self.physics = QuantumEligibilityPhysics(isotope=isotope)
        self._T2 = self.physics.get_T2_effective()
        self._decay_rate = self.physics.get_decay_rate()
    
    def activate(self, strength: float = 1.0):
        """
        Synaptic activation - creates eligibility trace
        
        This represents:
        - Calcium influx
        - Dimer formation
        - Quantum state preparation
        
        Args:
            strength: Activation strength (0-1), affects eligibility created
        """
        strength = np.clip(strength, 0.0, 1.0)
        
        # Create eligibility (saturates at max)
        new_eligibility = self.state.eligibility + strength
        self.state.eligibility = min(new_eligibility, self.params.eligibility_saturation)
        
        # Prepare quantum state (singlet probability increases)
        # Rate depends on how much new eligibility was actually added
        delta_e = self.state.eligibility - (new_eligibility - strength)
        singlet_boost = self.params.singlet_creation_rate * delta_e
        self.state.singlet_probability = min(1.0, 
            self.state.singlet_probability + singlet_boost * (1.0 - self.state.singlet_probability))
        
        # Timing
        self.state.time_since_activation = 0.0
        self.state.n_activations += 1
        
        # History
        if self._history_enabled and 'activation' in self._history:
            self._history['activation'].append(strength)
    
    def step(self, dt: float):
        """
        Time evolution - eligibility decays with T2
        
        This is the quantum decoherence process:
        - Nuclear spins lose phase coherence
        - Singlet probability decays toward thermal equilibrium
        - Eligibility trace decays exponentially
        
        Args:
            dt: Time step in seconds
        """
        # Exponential decay with physics-derived T2
        decay_factor = np.exp(-dt / self._T2)
        
        # Eligibility decay
        self.state.eligibility *= decay_factor
        
        # Singlet probability decays toward thermal equilibrium
        thermal = self.params.singlet_thermal
        self.state.singlet_probability = thermal + (self.state.singlet_probability - thermal) * decay_factor
        
        # Update timing
        if self.state.time_since_activation < np.inf:
            self.state.time_since_activation += dt
        
        # History
        if self._history_enabled:
            if 'eligibility' in self._history:
                self._history['eligibility'].append(self.state.eligibility)
            if 'singlet_prob' in self._history:
                self._history['singlet_prob'].append(self.state.singlet_probability)
    
    def apply_reward(self, reward: float, learning_rate: Optional[float] = None) -> float:
        """
        Apply reward signal - converts eligibility to weight change
        
        This is the "measurement" that collapses quantum exploration.
        Dopamine arrival triggers this in biology.
        
        The weight change is:
            Δw = learning_rate × reward × eligibility
        
        After reward, eligibility is partially consumed (measurement collapses state).
        
        Args:
            reward: Reward signal (can be positive or negative)
            learning_rate: Override default learning rate
            
        Returns:
            Actual weight change applied
        """
        lr = learning_rate if learning_rate is not None else self.params.learning_rate
        
        # Only apply if eligible
        if not self.is_eligible:
            return 0.0
        
        # Weight change proportional to eligibility
        delta_w = lr * reward * self.state.eligibility
        
        # Apply with bounds
        new_weight = np.clip(
            self.state.weight + delta_w,
            self.params.weight_min,
            self.params.weight_max
        )
        actual_delta = new_weight - self.state.weight
        self.state.weight = new_weight
        
        # Measurement partially collapses eligibility
        # (In biology: dopamine-dependent plasticity consumes the trace)
        self.state.eligibility *= 0.5
        
        # Singlet probability also decays (measurement disturbs quantum state)
        self.state.singlet_probability = self.params.singlet_thermal + \
            (self.state.singlet_probability - self.params.singlet_thermal) * 0.5
        
        # Statistics
        self.state.n_weight_updates += 1
        self.state.cumulative_weight_change += abs(actual_delta)
        
        # History
        if self._history_enabled and 'weight' in self._history:
            self._history['weight'].append(self.state.weight)
        
        return actual_delta
    
    def get_value(self, input_signal: float = 1.0) -> float:
        """
        Get output value (weight × input)
        
        Args:
            input_signal: Input to synapse (default 1.0)
            
        Returns:
            Weighted output
        """
        return self.state.weight * input_signal
    
    def enable_history(self, variables: Optional[List[str]] = None):
        """
        Enable history tracking for analysis
        
        Args:
            variables: List of variables to track, or None for defaults
        """
        self._history_enabled = True
        if variables is None:
            variables = ['eligibility', 'singlet_prob', 'weight', 'activation']
        self._history = {v: [] for v in variables}
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get recorded history as numpy arrays"""
        return {k: np.array(v) for k, v in self._history.items()}
    
    def get_state_dict(self) -> Dict:
        """Get complete state for serialization"""
        return {
            'isotope': self.physics.isotope.value,
            'T2': self._T2,
            'weight': self.state.weight,
            'eligibility': self.state.eligibility,
            'singlet_probability': self.state.singlet_probability,
            'n_activations': self.state.n_activations,
            'n_weight_updates': self.state.n_weight_updates,
        }
    
    def __repr__(self) -> str:
        return (f"QuantumSynapse(isotope={self.physics.isotope.value}, "
                f"T2={self._T2:.1f}s, w={self.state.weight:.3f}, "
                f"e={self.state.eligibility:.3f})")


# =============================================================================
# SYNAPSE POPULATION (for network-level analysis)
# =============================================================================

class SynapsePopulation:
    """
    Collection of quantum synapses for population-level analysis
    
    Provides vectorized operations for efficiency.
    """
    
    def __init__(self, 
                 n_synapses: int,
                 isotope: Isotope = Isotope.P31,
                 seed: Optional[int] = None):
        """
        Create population of synapses
        
        Args:
            n_synapses: Number of synapses
            isotope: Isotope for all synapses
            seed: Random seed
        """
        self.n_synapses = n_synapses
        self.rng = np.random.default_rng(seed)
        
        # Create synapses
        seeds = self.rng.integers(0, 2**31, size=n_synapses)
        self.synapses = [
            QuantumSynapse(isotope=isotope, seed=int(s))
            for s in seeds
        ]
        
        # Cache physics (same for all)
        self.physics = self.synapses[0].physics
        self.T2 = self.synapses[0].T2
    
    def activate_subset(self, indices: np.ndarray, strengths: Optional[np.ndarray] = None):
        """Activate subset of synapses"""
        if strengths is None:
            strengths = np.ones(len(indices))
        
        for idx, strength in zip(indices, strengths):
            self.synapses[idx].activate(strength)
    
    def step_all(self, dt: float):
        """Time step for all synapses"""
        for syn in self.synapses:
            syn.step(dt)
    
    def apply_reward_all(self, reward: float) -> np.ndarray:
        """Apply reward to all synapses, return weight changes"""
        return np.array([syn.apply_reward(reward) for syn in self.synapses])
    
    def get_eligibilities(self) -> np.ndarray:
        """Get all eligibility values"""
        return np.array([syn.eligibility for syn in self.synapses])
    
    def get_weights(self) -> np.ndarray:
        """Get all weights"""
        return np.array([syn.weight for syn in self.synapses])
    
    def set_weights(self, weights: np.ndarray):
        """Set all weights"""
        for syn, w in zip(self.synapses, weights):
            syn.state.weight = w


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM SYNAPSE MODULE TEST")
    print("=" * 60)
    
    # Test 1: Basic synapse behavior
    print("\n--- Test 1: Basic Synapse ---")
    syn = QuantumSynapse(isotope=Isotope.P31)
    print(f"Initial: {syn}")
    print(f"T2 = {syn.T2:.1f} seconds")
    
    # Activate
    syn.activate(strength=1.0)
    print(f"After activation: eligibility={syn.eligibility:.3f}, singlet={syn.singlet_prob:.3f}")
    
    # Let time pass
    for t in [1.0, 5.0, 10.0, 25.0]:
        syn_test = QuantumSynapse(isotope=Isotope.P31)
        syn_test.activate(strength=1.0)
        syn_test.step(dt=t)
        print(f"  After {t:.0f}s: eligibility={syn_test.eligibility:.3f}")
    
    # Test 2: Isotope comparison
    print("\n--- Test 2: Isotope Comparison ---")
    for isotope in Isotope:
        syn = QuantumSynapse(isotope=isotope)
        syn.activate(strength=1.0)
        
        # Measure decay after 10 seconds
        syn.step(dt=10.0)
        
        print(f"{isotope.value}: T2={syn.T2:.1f}s, eligibility after 10s = {syn.eligibility:.4f}")
    
    # Test 3: Learning with delayed reward
    print("\n--- Test 3: Delayed Reward Learning ---")
    syn = QuantumSynapse(isotope=Isotope.P31)
    syn.enable_history()
    
    # Simulate: activate, wait, then reward
    syn.activate(strength=1.0)
    
    # Wait 30 seconds (within T2 window)
    for _ in range(300):  # 300 × 0.1s = 30s
        syn.step(dt=0.1)
    
    print(f"Eligibility after 30s delay: {syn.eligibility:.4f}")
    
    # Apply reward
    delta_w = syn.apply_reward(reward=1.0)
    print(f"Weight change: {delta_w:.4f}")
    print(f"New weight: {syn.weight:.4f}")
    
    # Compare with P32 (should fail)
    syn_p32 = QuantumSynapse(isotope=Isotope.P32)
    syn_p32.activate(strength=1.0)
    for _ in range(300):
        syn_p32.step(dt=0.1)
    
    print(f"\nP32 eligibility after 30s: {syn_p32.eligibility:.4f}")
    delta_w_p32 = syn_p32.apply_reward(reward=1.0)
    print(f"P32 weight change: {delta_w_p32:.4f}")
    
    print("\n--- Conclusion ---")
    print(f"P31 learned (Δw={delta_w:.4f}), P32 did not (Δw={delta_w_p32:.4f})")
    print("This demonstrates the isotope effect on temporal credit assignment!")