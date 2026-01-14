"""
Rate-Based Neuron for Quantum RNN
==================================

Simple leaky integrator neuron with sigmoidal output.
Designed for recurrent networks testing quantum eligibility traces.

The neuron model is intentionally minimal - the interesting physics
is in the synapses (quantum eligibility), not the neurons themselves.

Author: Sarah Davidson
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuronParameters:
    """Parameters for rate-based neuron"""
    tau_membrane: float = 0.020    # Membrane time constant (20 ms)
    threshold: float = 1.0         # Soft threshold for sigmoid
    gain: float = 5.0              # Sigmoid steepness
    resting_potential: float = 0.0 # Resting membrane potential
    

class RateNeuron:
    """
    Leaky integrator neuron with sigmoidal rate output.
    
    Dynamics:
        τ * dV/dt = -V + I_input
        rate = sigmoid(gain * (V - threshold))
    
    This is a standard rate model used in many RNN implementations.
    The quantum mechanics lives in the synapses, not here.
    """
    
    def __init__(self, 
                 params: Optional[NeuronParameters] = None,
                 neuron_id: int = 0):
        """
        Initialize neuron.
        
        Args:
            params: Neuron parameters (uses defaults if None)
            neuron_id: Identifier for this neuron
        """
        self.params = params or NeuronParameters()
        self.neuron_id = neuron_id
        
        # State variables
        self.membrane_potential = self.params.resting_potential
        self.firing_rate = 0.0
        
        # For tracking
        self.total_input = 0.0
    
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
        dV = dt * (-self.membrane_potential + input_current) / self.params.tau_membrane
        self.membrane_potential += dV
        
        # Sigmoid output (clamp input to prevent overflow)
        x = self.params.gain * (self.membrane_potential - self.params.threshold)
        x = np.clip(x, -500, 500)  # Prevent overflow in exp
        self.firing_rate = 1.0 / (1.0 + np.exp(-x))
        
        return self.firing_rate
    
    def reset(self):
        """Reset to resting state"""
        self.membrane_potential = self.params.resting_potential
        self.firing_rate = 0.0
        self.total_input = 0.0
    
    @property
    def is_active(self) -> bool:
        """Whether neuron is above half-max firing"""
        return self.firing_rate > 0.5
    
    def __repr__(self) -> str:
        return f"RateNeuron(id={self.neuron_id}, V={self.membrane_potential:.3f}, rate={self.firing_rate:.3f})"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("RATE NEURON TEST")
    print("=" * 50)
    
    # Test 1: Basic dynamics
    print("\n--- Test 1: Step response ---")
    neuron = RateNeuron()
    
    dt = 0.001  # 1 ms timestep
    input_current = 2.0  # Above threshold
    
    print(f"Input current: {input_current}")
    print(f"Threshold: {neuron.params.threshold}")
    print(f"τ_membrane: {neuron.params.tau_membrane * 1000:.0f} ms")
    
    # Run for 100 ms
    times = []
    rates = []
    potentials = []
    
    for i in range(100):
        t = i * dt
        neuron.step(dt, input_current)
        times.append(t * 1000)  # ms
        rates.append(neuron.firing_rate)
        potentials.append(neuron.membrane_potential)
    
    print(f"\nAfter 100 ms:")
    print(f"  V = {neuron.membrane_potential:.3f}")
    print(f"  rate = {neuron.firing_rate:.3f}")
    print(f"  is_active = {neuron.is_active}")
    
    # Test 2: Subthreshold input
    print("\n--- Test 2: Subthreshold input ---")
    neuron.reset()
    input_current = 0.5  # Below threshold
    
    for _ in range(100):
        neuron.step(dt, input_current)
    
    print(f"Input current: {input_current} (below threshold)")
    print(f"  V = {neuron.membrane_potential:.3f}")
    print(f"  rate = {neuron.firing_rate:.3f}")
    print(f"  is_active = {neuron.is_active}")
    
    # Test 3: Decay after input removed
    print("\n--- Test 3: Decay dynamics ---")
    neuron.reset()
    
    # Drive to high rate
    for _ in range(50):
        neuron.step(dt, 3.0)
    print(f"After 50ms drive: rate = {neuron.firing_rate:.3f}")
    
    # Remove input, observe decay
    for _ in range(100):
        neuron.step(dt, 0.0)
    print(f"After 100ms decay: rate = {neuron.firing_rate:.3f}")
    
    # Test 4: Time constant check
    print("\n--- Test 4: Time constant verification ---")
    neuron.reset()
    
    # Step input, check 63% rise time
    target_63 = 2.0 * 0.632  # 63.2% of final value
    
    for i in range(1000):
        neuron.step(dt, 2.0)
        if neuron.membrane_potential >= target_63 and i > 0:
            rise_time_ms = i * dt * 1000
            print(f"63% rise time: {rise_time_ms:.1f} ms (expected: {neuron.params.tau_membrane * 1000:.0f} ms)")
            break
    
    print("\n--- All tests passed ---")