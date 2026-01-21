"""
Rate-Based Neuron for Quantum RNN
==================================

Simple leaky integrator neuron with sigmoidal output.
The neuron model is intentionally minimal - the interesting physics
is in the Model6 quantum synapses, not the neurons themselves.

Dynamics:
    τ_m * dV/dt = -V + I_total
    rate = sigmoid(gain * (V - threshold))

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuronConfig:
    """Configuration for rate-based neuron"""
    tau_membrane: float = 0.020      # Membrane time constant (20 ms)
    threshold: float = 1.0           # Soft threshold for sigmoid
    gain: float = 5.0                # Sigmoid steepness  
    resting_potential: float = 0.0   # Resting membrane potential
    rate_noise: float = 0.0          # Optional noise on firing rate


class RateNeuron:
    """
    Leaky integrator neuron with sigmoidal rate output.
    
    This is a standard rate model. The quantum mechanics lives in 
    the synapses (Model6QuantumSynapse), not here.
    
    Usage:
        neuron = RateNeuron(neuron_id=0)
        
        for t in range(n_steps):
            total_input = external + recurrent
            rate = neuron.step(dt, total_input)
    """
    
    def __init__(self, 
                 config: Optional[NeuronConfig] = None,
                 neuron_id: int = 0):
        """
        Initialize neuron.
        
        Args:
            config: Neuron parameters (uses defaults if None)
            neuron_id: Identifier for this neuron
        """
        self.config = config or NeuronConfig()
        self.neuron_id = neuron_id
        
        # State variables
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0
        self.total_input = 0.0
        
        # For RNG if using noise
        self._rng = np.random.default_rng()
    
    def step(self, dt: float, input_current: float) -> float:
        """
        Update neuron state for one timestep.
        
        Args:
            dt: Timestep in seconds
            input_current: Total input (external + recurrent)
            
        Returns:
            Current firing rate [0, 1]
        """
        self.total_input = input_current
        
        # Leaky integration: τ * dV/dt = -V + I
        dV = dt * (-self.membrane_potential + input_current) / self.config.tau_membrane
        self.membrane_potential += dV
        
        # Sigmoid output (clamp to prevent overflow)
        x = self.config.gain * (self.membrane_potential - self.config.threshold)
        x = np.clip(x, -500, 500)
        self.firing_rate = 1.0 / (1.0 + np.exp(-x))
        
        # Optional noise
        if self.config.rate_noise > 0:
            noise = self._rng.normal(0, self.config.rate_noise)
            self.firing_rate = np.clip(self.firing_rate + noise, 0, 1)
        
        return self.firing_rate
    
    def reset(self):
        """Reset to resting state"""
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0
        self.total_input = 0.0
    
    @property
    def is_active(self) -> bool:
        """Whether neuron is above half-max firing"""
        return self.firing_rate > 0.5
    
    def __repr__(self) -> str:
        return (f"RateNeuron(id={self.neuron_id}, "
                f"V={self.membrane_potential:.3f}, "
                f"rate={self.firing_rate:.3f})")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RATE NEURON TEST")
    print("=" * 50)
    
    neuron = RateNeuron()
    dt = 0.001  # 1 ms
    
    # Test step response
    print("\n--- Step response to input=2.0 ---")
    for i in range(100):
        neuron.step(dt, 2.0)
        if i % 20 == 0:
            print(f"  t={i}ms: V={neuron.membrane_potential:.3f}, rate={neuron.firing_rate:.3f}")
    
    print(f"\nFinal: V={neuron.membrane_potential:.3f}, rate={neuron.firing_rate:.3f}")
    print(f"Active: {neuron.is_active}")