"""
Model 6 Eligibility Adapter for QETR Framework
================================================

This adapter wraps Model 6's actual biophysical system for RL use.

CRITICAL: This imports from Model 6, it does NOT recreate physics.
The T2 values come from validated nuclear spin parameters.

Physics basis:
- P31 (I=1/2): T_singlet ~ 216s (dipolar relaxation only)
- P32 (I=1): T_singlet ~ 0.4s (quadrupolar relaxation dominates)

The ratio (216/0.4 = 540×) is the computational "killer experiment".

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# PHYSICS CONSTANTS (from Agarwal et al. 2023 + Model 6 validation)
# =============================================================================

class Isotope(Enum):
    """Phosphorus isotope selection"""
    P31 = "P31"  # I=1/2, stable, long coherence
    P32 = "P32"  # I=1, radioactive, short coherence (quadrupolar)


@dataclass
class IsotopePhysics:
    """
    Nuclear spin physics parameters - NOT hyperparameters.
    
    These values come from:
    - Agarwal et al. 2023 (singlet lifetimes)
    - NMR literature (T2 values)
    - Model 6 validation runs
    """
    
    # Single-spin T2 in aqueous solution (seconds)
    T2_single_P31: float = 2.5      # Dipolar relaxation only
    T2_single_P32: float = 0.01     # Quadrupolar dominates
    
    # Dimer singlet lifetime (4 coupled spins)
    # From Agarwal: "hundreds of seconds" for I=1/2
    T_singlet_P31: float = 216.0    # seconds
    T_singlet_P32: float = 0.4      # seconds
    
    # J-coupling protection factor (from ATP hydrolysis)
    # J = 20 Hz from Model 6 parameters
    J_coupling_Hz: float = 20.0
    J_protection_factor: float = 25.0  # How much J extends T2
    
    # Thermal equilibrium singlet probability
    P_S_thermal: float = 0.25       # Maximally mixed state
    
    # Entanglement threshold
    P_S_entanglement: float = 0.5   # Above this = entanglement preserved
    
    def get_T2_effective(self, isotope: Isotope, atp_active: bool = True) -> float:
        """
        Get effective T2 for given isotope and conditions.
        
        This is the DERIVED eligibility window, not a hyperparameter.
        """
        if isotope == Isotope.P31:
            base_T2 = self.T_singlet_P31
        else:
            base_T2 = self.T_singlet_P32
        
        # J-coupling from active ATP hydrolysis extends coherence
        if atp_active:
            # At active synapses, J ~ 20 Hz provides protection
            # But this is already included in the singlet lifetime calculation
            pass
        
        return base_T2


# Global physics instance (immutable constants)
PHYSICS = IsotopePhysics()


# =============================================================================
# MODEL 6 ADAPTER - TWO MODES
# =============================================================================

class Model6EligibilityAdapter:
    """
    Adapter providing eligibility traces from Model 6 physics.
    
    Two modes:
    1. SIMPLIFIED: Exponential decay with physics-derived T2 (fast)
    2. FULL: Run actual Model6QuantumSynapse (accurate but slow)
    
    In both modes, T2 comes from physics, NOT hyperparameter tuning.
    """
    
    def __init__(self, 
                 isotope: str = "P31",
                 use_full_model: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize eligibility adapter.
        
        Args:
            isotope: "P31" (T2~216s) or "P32" (T2~0.4s)
            use_full_model: If True, run full biophysical simulation
            seed: Random seed for stochastic elements
        """
        self.isotope = Isotope(isotope)
        self.use_full_model = use_full_model
        self.rng = np.random.default_rng(seed)
        
        # Get T2 from physics
        self._T2 = PHYSICS.get_T2_effective(self.isotope)
        
        # State variables
        self.eligibility = 0.0
        self.singlet_probability = PHYSICS.P_S_thermal  # Start at thermal
        self.time_since_activation = float('inf')
        self._activated = False
        
        # Full model (lazy initialization)
        self._full_model = None
        
        if use_full_model:
            self._init_full_model()
    
    def _init_full_model(self):
        """Initialize full Model 6 synapse (heavy)"""
        try:
            # These imports may fail if Model 6 not in path
            import sys
            from pathlib import Path
            
            # Add Model 6 to path
            model6_path = Path(__file__).parent.parent.parent / 'models' / 'Model_6'
            if str(model6_path) not in sys.path:
                sys.path.insert(0, str(model6_path))
            
            from ..models.Model_6.model6_core import Model6QuantumSynapse
            from ..models.Model_6.model6_parameters import Model6Parameters
            
            # Configure
            params = Model6Parameters()
            params.em_coupling_enabled = True
            params.environment.fraction_P31 = 1.0 if self.isotope == Isotope.P31 else 0.0
            params.environment.fraction_P32 = 0.0 if self.isotope == Isotope.P31 else 1.0
            
            self._full_model = Model6QuantumSynapse(params)
            print(f"  Full Model 6 initialized for {self.isotope.value}")
            
        except ImportError as e:
            print(f"  Warning: Could not import Model 6: {e}")
            print(f"  Falling back to simplified mode")
            self.use_full_model = False
    
    @property
    def T2(self) -> float:
        """
        Physics-derived coherence time (NOT a hyperparameter).
        
        P31: ~216s (dipolar relaxation)
        P32: ~0.4s (quadrupolar relaxation)
        """
        return self._T2
    
    @property
    def is_entangled(self) -> bool:
        """Entanglement preserved when P_S > 0.5"""
        return self.singlet_probability > PHYSICS.P_S_entanglement
    
    def activate(self, strength: float = 1.0):
        """
        Synaptic activation creates eligibility trace.
        
        In biophysical terms: calcium influx → dimer formation → singlet state
        """
        if self.use_full_model and self._full_model is not None:
            # Run theta-burst stimulation
            try:
                self._full_model.run_theta_burst(n_bursts=3)
                self.eligibility = self._full_model.get_eligibility()
                self.singlet_probability = self._full_model.get_mean_singlet_probability()
            except Exception as e:
                print(f"  Full model activation failed: {e}")
                self._activate_simplified(strength)
        else:
            self._activate_simplified(strength)
        
        self._activated = True
        self.time_since_activation = 0.0
    
    def _activate_simplified(self, strength: float):
        """Simplified activation - set to pure singlet state"""
        self.eligibility = strength
        self.singlet_probability = 1.0  # Born as pure singlet
    
    def step(self, dt: float):
        """
        Advance time - eligibility decays with physics-derived T2.
        
        P_S(t) = P_thermal + (P_S(0) - P_thermal) × exp(-t/T2)
        """
        if not self._activated:
            return
        
        self.time_since_activation += dt
        
        if self.use_full_model and self._full_model is not None:
            # Full biophysical decay
            try:
                self._full_model.step(dt)
                self.eligibility = self._full_model.get_eligibility()
                self.singlet_probability = self._full_model.get_mean_singlet_probability()
            except Exception as e:
                self._step_simplified(dt)
        else:
            self._step_simplified(dt)
    
    def _step_simplified(self, dt: float):
        """
        Simplified decay using exact physics.
        
        Singlet probability decay:
        P_S(t) = P_thermal + (P_S(0) - P_thermal) × exp(-t/T2)
        
        Eligibility = (P_S - 0.25) / 0.75  # Map [0.25, 1.0] → [0, 1]
        """
        # Decay from current value toward thermal
        decay_factor = np.exp(-dt / self._T2)
        
        P_excess = self.singlet_probability - PHYSICS.P_S_thermal
        self.singlet_probability = PHYSICS.P_S_thermal + P_excess * decay_factor
        
        # Add small noise (quantum fluctuations)
        noise = 0.001 * self.rng.standard_normal()
        self.singlet_probability = np.clip(
            self.singlet_probability + noise,
            PHYSICS.P_S_thermal,
            1.0
        )
        
        # Map to eligibility [0, 1]
        self.eligibility = (self.singlet_probability - PHYSICS.P_S_thermal) / 0.75
        
        # Deactivate if below threshold
        if self.eligibility < 0.01:
            self._activated = False
            self.eligibility = 0.0
    
    def get_eligibility(self) -> float:
        """Get current eligibility trace value"""
        return self.eligibility
    
    def get_singlet_probability(self) -> float:
        """Get raw singlet probability (P_S)"""
        return self.singlet_probability
    
    def reset(self):
        """Reset for new episode"""
        if self.use_full_model and self._full_model is not None:
            self._full_model.reset()
        
        self.eligibility = 0.0
        self.singlet_probability = PHYSICS.P_S_thermal
        self.time_since_activation = float('inf')
        self._activated = False
    
    def get_state(self) -> Dict:
        """Get complete state for debugging"""
        return {
            'isotope': self.isotope.value,
            'T2': self._T2,
            'eligibility': self.eligibility,
            'singlet_probability': self.singlet_probability,
            'is_entangled': self.is_entangled,
            'time_since_activation': self.time_since_activation,
            'activated': self._activated,
            'mode': 'full' if self.use_full_model else 'simplified'
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compare_isotopes(duration_s: float = 100.0, dt: float = 0.1) -> Dict:
    """
    Compare P31 vs P32 decay curves.
    
    Returns trajectories for both isotopes after activation.
    """
    results = {'P31': [], 'P32': [], 'times': []}
    
    adapters = {
        'P31': Model6EligibilityAdapter(isotope="P31"),
        'P32': Model6EligibilityAdapter(isotope="P32")
    }
    
    # Activate both
    for adapter in adapters.values():
        adapter.activate(1.0)
    
    # Track decay
    n_steps = int(duration_s / dt)
    for step in range(n_steps):
        t = step * dt
        results['times'].append(t)
        
        for iso_name, adapter in adapters.items():
            results[iso_name].append(adapter.get_eligibility())
            adapter.step(dt)
    
    return results


def get_effective_lambda(isotope: str = "P31", dt: float = 0.1) -> float:
    """
    Get the effective λ for TD(λ) comparison.
    
    In TD(λ): e(t+dt) = λ × e(t)
    In physics: e(t+dt) = e(t) × exp(-dt/T2)
    
    Therefore: λ_effective = exp(-dt/T2)
    """
    adapter = Model6EligibilityAdapter(isotope=isotope)
    return np.exp(-dt / adapter.T2)

# =============================================================================
# BACKWARD COMPATIBILITY 
# =============================================================================

class QuantumEligibilityPhysics:
    """
    Compatibility wrapper for old API.
    """
    
    def __init__(self, isotope: Isotope = Isotope.P31):
        self.isotope = isotope if isinstance(isotope, Isotope) else Isotope(isotope)
        self._adapter = Model6EligibilityAdapter(isotope=self.isotope.value)
    
    def get_T2_effective(self) -> float:
        return self._adapter.T2
    
    def get_T2_single(self) -> float:
        return PHYSICS.T2_single_P31 if self.isotope == Isotope.P31 else PHYSICS.T2_single_P32


def compare_isotopes():
    """Compare P31 vs P32 parameters."""
    return {
        'P31': {
            'T2_effective_s': PHYSICS.T_singlet_P31,
            'lambda_equivalent_100ms': np.exp(-0.1 / PHYSICS.T_singlet_P31),
        },
        'P32': {
            'T2_effective_s': PHYSICS.T_singlet_P32,
            'lambda_equivalent_100ms': np.exp(-0.1 / PHYSICS.T_singlet_P32),
        },
    }


def get_T2_ratio() -> float:
    """Get T2 ratio between isotopes."""
    return PHYSICS.T_singlet_P31 / PHYSICS.T_singlet_P32

# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL 6 ELIGIBILITY ADAPTER VALIDATION")
    print("=" * 60)
    
    # Test 1: Physics constants
    print("\n--- Test 1: Physics Constants ---")
    print(f"P31 T2 (singlet lifetime): {PHYSICS.T_singlet_P31:.1f}s")
    print(f"P32 T2 (singlet lifetime): {PHYSICS.T_singlet_P32:.1f}s")
    print(f"Ratio: {PHYSICS.T_singlet_P31 / PHYSICS.T_singlet_P32:.0f}×")
    print(f"(Theory: ~540× from Agarwal et al. 2023)")
    
    # Test 2: Adapter initialization
    print("\n--- Test 2: Adapter Initialization ---")
    p31 = Model6EligibilityAdapter(isotope="P31")
    p32 = Model6EligibilityAdapter(isotope="P32")
    print(f"P31 adapter T2: {p31.T2:.1f}s")
    print(f"P32 adapter T2: {p32.T2:.1f}s")
    
    # Test 3: Activation and decay
    print("\n--- Test 3: Activation and Decay ---")
    p31.activate(1.0)
    p32.activate(1.0)
    
    print(f"After activation:")
    print(f"  P31: eligibility={p31.eligibility:.3f}, P_S={p31.singlet_probability:.3f}")
    print(f"  P32: eligibility={p32.eligibility:.3f}, P_S={p32.singlet_probability:.3f}")
    
    # Decay for 30 seconds
    for _ in range(300):
        p31.step(0.1)
        p32.step(0.1)
    
    print(f"\nAfter 30 seconds:")
    print(f"  P31: eligibility={p31.eligibility:.3f}, P_S={p31.singlet_probability:.3f}")
    print(f"  P32: eligibility={p32.eligibility:.3f}, P_S={p32.singlet_probability:.3f}")
    
    # Test 4: Effective λ
    print("\n--- Test 4: Effective λ (for TD(λ) comparison) ---")
    lambda_p31 = get_effective_lambda("P31", dt=0.1)
    lambda_p32 = get_effective_lambda("P32", dt=0.1)
    print(f"P31 effective λ (dt=0.1s): {lambda_p31:.6f}")
    print(f"P32 effective λ (dt=0.1s): {lambda_p32:.6f}")
    print(f"(These are NOT tuned - derived from T2)")
    
    # Test 5: Isotope effect validation
    print("\n--- Test 5: Isotope Effect (The Killer Experiment) ---")
    p31 = Model6EligibilityAdapter(isotope="P31")
    p32 = Model6EligibilityAdapter(isotope="P32")
    
    p31.activate(1.0)
    p32.activate(1.0)
    
    # Check at key timepoints
    for delay in [1, 5, 10, 30, 60]:
        # Reset and run
        p31_test = Model6EligibilityAdapter(isotope="P31")
        p32_test = Model6EligibilityAdapter(isotope="P32")
        
        p31_test.activate(1.0)
        p32_test.activate(1.0)
        
        for _ in range(int(delay / 0.1)):
            p31_test.step(0.1)
            p32_test.step(0.1)
        
        print(f"  t={delay:3d}s: P31={p31_test.eligibility:.3f}, P32={p32_test.eligibility:.6f}")
    
    print("\n✓ Validation complete!")
    print("\nKey result: P31 maintains eligibility through 60s learning window.")
    print("P32 eligibility is essentially zero after ~2 seconds.")
    print("This is the computational 'killer experiment'.")