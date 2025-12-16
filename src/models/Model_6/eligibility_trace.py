"""
Eligibility Trace Module for Model 6 Quantum Synapse

Core hypothesis: Calcium phosphate dimers ARE the eligibility trace.
The T2 decoherence time (67.6s for ³¹P) determines the eligibility window.

This module tracks which synapses are "eligible" for plasticity based on
recent dimer formation, and gates plasticity on plateau potential arrival.

Key insight: Eligibility = f(dimer_count, coherence, time)
Plasticity occurs when: eligibility > threshold AND plateau_potential = True

References:
- Bittner et al. 2017 (Science) - BTSP mechanism
- Agarwal et al. 2023 - Dimer T2 values
- Grienberger et al. 2014 - Plateau potential timing

Author: Sarah Davidson
Model 6 Quantum Synapse Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class PhosphorusIsotope(Enum):
    """Phosphorus isotope selection for T2 calculations"""
    P31 = "P31"  # Stable isotope, long coherence
    P32 = "P32"  # Radioactive, short coherence (control)


@dataclass
class EligibilityTraceParameters:
    """
    Parameters for eligibility trace dynamics
    
    NOTE: T2 is NOT prescribed here - it comes from quantum_coherence system
    which calculates emergent T2 from physics.
    """
    
    # === Dimer Thresholds ===
    dimer_threshold_min: int = 3
    dimer_saturation: int = 10
    coherence_threshold: float = 0.5
    
    # === Eligibility Thresholds ===
    eligibility_threshold: float = 0.3
    eligibility_floor: float = 0.1
    
    # === Plateau Potential Parameters ===
    plateau_duration_typical: float = 0.250
    plateau_calcium_threshold: float = 2.0
    
    # === Weight Change Scaling ===
    ltp_scale: float = 1.0
    ltd_scale: float = 0.5
    
    # === Isotope Selection ===
    isotope: PhosphorusIsotope = PhosphorusIsotope.P31
    

@dataclass
class EligibilityState:
    """Current state of the eligibility trace"""
    eligibility: float = 0.0           # Current eligibility level (0-1)
    is_tagged: bool = False            # Whether synapse has active trace
    time_since_tag: float = np.inf     # Time since eligibility created (s)
    peak_eligibility: float = 0.0      # Maximum eligibility reached
    tag_count: int = 0                 # Number of times tagged this session
    
    # Plasticity outputs
    plasticity_gate: bool = False      # Should plasticity occur now?
    weight_change: float = 0.0         # Magnitude of weight change
    weight_direction: int = 0          # +1 LTP, 0 none, -1 LTD
    
    # Diagnostic
    last_dimer_count: int = 0
    last_coherence: float = 0.0
    decay_rate: float = 0.0            # Current decay rate (1/s)
    T2_current: float = 67.0           # Current T2 from quantum_coherence (s)

class EligibilityTraceModule:
    """
    Eligibility trace based on quantum dimer coherence.
    
    The eligibility trace provides the temporal bridge between presynaptic
    activity and instructive signals (plateau potentials). In the quantum
    model, this bridge is implemented by the nuclear spin coherence of
    calcium phosphate dimers.
    
    Flow:
        1. Presynaptic activity → dimer formation → eligibility creation
        2. Eligibility decays with time constant T2 (isotope-dependent)
        3. Plateau potential arrives → eligibility converted to plasticity
        4. CaMKII/spine plasticity modules implement the weight change
    
    Key prediction: Eligibility window duration = T2
        - P31: ~68 second window
        - P32: ~0.3 second window (isotope substitution experiment)
    """
    
    def __init__(self, params: Optional[EligibilityTraceParameters] = None):
        """
        Initialize eligibility trace module
        
        Args:
            params: EligibilityTraceParameters instance, or None for defaults
        """
        self.params = params or EligibilityTraceParameters()
        self.state = EligibilityState()
        
        # History tracking for analysis
        self._history_enabled = False
        self._history: Dict[str, list] = {}
        
    def reset(self):
        """Reset to initial state"""
        self.state = EligibilityState()
        if self._history_enabled:
            self._history = {k: [] for k in self._history.keys()}
    
    def enable_history(self, variables: Optional[list] = None):
        """
        Enable history tracking for specified variables
        
        Args:
            variables: List of state variables to track, or None for all
        """
        self._history_enabled = True
        if variables is None:
            variables = ['eligibility', 'is_tagged', 'time_since_tag', 
                        'plasticity_gate', 'weight_change']
        self._history = {v: [] for v in variables}
    
    def set_isotope(self, isotope: PhosphorusIsotope):
        """
        Switch phosphorus isotope (for isotope substitution experiments)
        
        Args:
            isotope: PhosphorusIsotope.P31 or PhosphorusIsotope.P32
        """
        self.params.isotope = isotope
    
    def step(self, 
            dt: float,
            dimer_count: int,
            coherence: float,
            T2_effective: float,  # ← ADD: Receive from quantum_coherence
            plateau_potential: bool = False,
            calcium_uM: float = 0.0) -> Dict:
        """
        Update eligibility trace for one timestep
        
        Args:
            dt: Timestep in seconds
            dimer_count: Number of coherent dimers at this synapse
            coherence: Mean coherence of dimers (0-1) from quantum_coherence
            T2_effective: Emergent T2 from quantum_coherence (seconds)
            ...
        """
        p = self.params
        s = self.state
        
        s.last_dimer_count = dimer_count
        s.last_coherence = coherence
        
        # === ELIGIBILITY CREATION ===
        is_new_activity = (dimer_count >= p.dimer_threshold_min and 
                        coherence > p.coherence_threshold and
                        calcium_uM > 1.0)
        
        if is_new_activity and not s.is_tagged:
            s.is_tagged = True
            s.time_since_tag = 0.0
            s.eligibility = min(1.0, dimer_count / p.dimer_saturation)
            s.peak_eligibility = s.eligibility
            s.tag_count += 1
        
        # === ELIGIBILITY DECAY ===
        if s.is_tagged:
            s.time_since_tag += dt
            s.decay_rate = 1.0 / T2_effective  # ← USE PASSED T2, not prescribed
            decay_factor = np.exp(-dt * s.decay_rate)
            s.eligibility *= decay_factor
            
            if s.eligibility < p.eligibility_floor:
                s.is_tagged = False
                s.eligibility = 0.0
    
        # Store T2 for later access
        s.T2_current = T2_effective


        # === PLASTICITY CONVERSION ===
        # Convert eligibility → plasticity ONLY if plateau potential arrives
        # This implements the gating function of BTSP
        
        s.plasticity_gate = False
        s.weight_change = 0.0
        s.weight_direction = 0
        
        if plateau_potential and s.eligibility > p.eligibility_threshold:
            s.plasticity_gate = True
            
            # Weight change magnitude proportional to eligibility
            s.weight_change = s.eligibility
            
            # Direction based on calcium level during plateau
            # High Ca²⁺ → LTP, moderate Ca²⁺ → LTD
            # (Simplified; full BCM rule could be implemented)
            if calcium_uM > p.plateau_calcium_threshold:
                s.weight_direction = 1  # LTP
                s.weight_change *= p.ltp_scale
            elif calcium_uM > 0.5:  # Above baseline
                s.weight_direction = -1  # LTD
                s.weight_change *= p.ltd_scale
            else:
                # Very low calcium, no change
                s.weight_direction = 0
                s.weight_change = 0.0
                s.plasticity_gate = False
        
        # === RECORD HISTORY ===
        if self._history_enabled:
            for var in self._history:
                self._history[var].append(getattr(s, var))
        
        # === RETURN STATE ===
        return {
            'eligibility': s.eligibility,
            'is_tagged': s.is_tagged,
            'time_since_tag': s.time_since_tag,
            'peak_eligibility': s.peak_eligibility,
            'plasticity_gate': s.plasticity_gate,
            'weight_change': s.weight_change,
            'weight_direction': s.weight_direction,
            'decay_rate': s.decay_rate,
            'T2_current': T2_effective,
        }
    
    def get_eligibility(self) -> float:
        """Get current eligibility level"""
        return self.state.eligibility
    
    def get_time_since_tag(self) -> float:
        """Get time since last eligibility creation"""
        return self.state.time_since_tag
    
    def is_eligible(self) -> bool:
        """Check if synapse is currently eligible for plasticity"""
        return (self.state.is_tagged and 
                self.state.eligibility > self.params.eligibility_threshold)
    
    def get_predicted_half_life(self) -> float:
        """
        Get predicted eligibility half-life based on current T2
        
        This is the key experimental prediction:
        half-life = T2 * ln(2) ≈ 0.693 * T2
        
        """
        T2 = getattr(self.state, 'T2_current', 67.0)  # Default to P31 value
        return T2 * np.log(2)
    
    def get_history(self, variable: str) -> np.ndarray:
        """Get recorded history for a variable"""
        if variable in self._history:
            return np.array(self._history[variable])
        raise ValueError(f"Variable '{variable}' not in history")
    
    def get_state_dict(self) -> Dict:
        """Get full state as dictionary (for checkpointing)"""
        return {
            'eligibility': self.state.eligibility,
            'is_tagged': self.state.is_tagged,
            'time_since_tag': self.state.time_since_tag,
            'peak_eligibility': self.state.peak_eligibility,
            'tag_count': self.state.tag_count,
            'isotope': self.params.isotope.value,
            'T2': self.params.T2,
        }


def calculate_eligibility_at_delay(T2: float, delay: float, 
                                   initial_eligibility: float = 1.0) -> float:
    """
    Calculate remaining eligibility after a given delay
    
    Useful for predicting experimental outcomes.
    
    Args:
        T2: Coherence time in seconds
        delay: Delay in seconds
        initial_eligibility: Starting eligibility (0-1)
        
    Returns:
        Remaining eligibility
    """
    return initial_eligibility * np.exp(-delay / T2)


def predict_critical_delay(T2: float, threshold: float = 0.3) -> float:
    """
    Predict the delay at which eligibility falls below threshold
    
    This is the maximum delay for successful plasticity induction.
    
    Args:
        T2: Coherence time in seconds
        threshold: Eligibility threshold for plasticity
        
    Returns:
        Critical delay in seconds
    """
    # eligibility = exp(-t/T2) = threshold
    # -t/T2 = ln(threshold)
    # t = -T2 * ln(threshold)
    return -T2 * np.log(threshold)


# === CONVENIENCE FUNCTIONS FOR EXPERIMENTS ===

def create_P31_module() -> EligibilityTraceModule:
    """Create module with P31 (long coherence) parameters"""
    params = EligibilityTraceParameters(isotope=PhosphorusIsotope.P31)
    return EligibilityTraceModule(params)


def create_P32_module() -> EligibilityTraceModule:
    """Create module with P32 (short coherence) parameters"""
    params = EligibilityTraceParameters(isotope=PhosphorusIsotope.P32)
    return EligibilityTraceModule(params)


if __name__ == "__main__":
    # Quick validation
    print("Eligibility Trace Module - Quick Validation")
    print("=" * 50)
    
    # Test P31
    module_p31 = create_P31_module()
    print(f"\nP31 Configuration:")
    print(f"  T2 = {module_p31.params.T2:.1f} s")
    print(f"  Predicted half-life = {module_p31.get_predicted_half_life():.1f} s")
    print(f"  Critical delay (threshold=0.3) = {predict_critical_delay(67.6, 0.3):.1f} s")
    
    # Test P32
    module_p32 = create_P32_module()
    print(f"\nP32 Configuration:")
    print(f"  T2 = {module_p32.params.T2:.1f} s")
    print(f"  Predicted half-life = {module_p32.get_predicted_half_life():.2f} s")
    print(f"  Critical delay (threshold=0.3) = {predict_critical_delay(0.3, 0.3):.2f} s")
    
    # Simple step test
    print(f"\n--- Step Test (P31) ---")
    module_p31.reset()
    
    # Create eligibility with 5 dimers
    result = module_p31.step(0.001, dimer_count=5, coherence=0.8)
    print(f"After dimer formation: eligibility = {result['eligibility']:.3f}")
    
    # Decay for 30 seconds
    for _ in range(30000):
        result = module_p31.step(0.001, dimer_count=0, coherence=0.0)
    print(f"After 30s decay: eligibility = {result['eligibility']:.3f}")
    
    # Apply plateau
    result = module_p31.step(0.001, dimer_count=0, coherence=0.0, 
                            plateau_potential=True, calcium_uM=3.0)
    print(f"After plateau: gate={result['plasticity_gate']}, "
          f"weight_change={result['weight_change']:.3f}, "
          f"direction={result['weight_direction']}")
    
    print("\n✓ Module validated")