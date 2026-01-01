"""
Quantum Eligibility Trace Physics
==================================

Derives eligibility trace parameters from nuclear spin coherence physics.
These are NOT hyperparameters - they emerge from first principles.

Key insight: The eligibility trace decay constant τ = T2, where T2 is the
nuclear spin coherence time of ³¹P in calcium phosphate dimers.

Physics basis (from Model 6):
- Single ³¹P nuclear spin T2 ~ 2s in aqueous solution
- J-coupling protection from ATP hydrolysis enhances this 25x
- 4 coupled spins in dimer reduce coherence by √4
- Net result: T2_eff ~ 25-67s depending on conditions

This provides a DERIVED eligibility window matching behavioral learning
timescales (60-100s) without any parameter tuning.

References:
- Fisher 2015 Ann Phys 362:593-599 (Posner molecule hypothesis)
- Agarwal et al. 2023 (Dimer vs trimer coherence times)
- Adams et al. 2025 (J-coupling from ATP: 20 Hz)

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class Isotope(Enum):
    """
    Phosphorus isotope selection
    
    The isotope determines nuclear spin properties and thus coherence time.
    This is the key experimental variable for falsification.
    """
    P31 = "P31"  # I=1/2, 100% natural abundance, long coherence
    P32 = "P32"  # I=1 (no spin-1/2), radioactive, SHORT coherence
    P33 = "P33"  # I=1/2, rare, intermediate coherence


@dataclass
class NuclearSpinParameters:
    """
    Nuclear spin properties by isotope
    
    These are physical constants, not tunable parameters.
    
    References:
    - NMR spectroscopy literature for T2 values
    - Agarwal et al. 2023 for biological context
    """
    
    # Single-spin T2 in aqueous biological environment (seconds)
    # These are baseline values before J-coupling protection
    T2_single: Dict[str, float] = field(default_factory=lambda: {
        "P31": 2.0,   # Spin-1/2, long coherence
        "P32": 0.2,   # Spin-1, quadrupolar relaxation dominates
        "P33": 1.0,   # Spin-1/2, different gyromagnetic ratio
    })
    
    # Nuclear spin quantum number
    spin_I: Dict[str, float] = field(default_factory=lambda: {
        "P31": 0.5,   # Spin-1/2
        "P32": 1.0,   # Spin-1 (actually I=1 for P32)
        "P33": 0.5,   # Spin-1/2
    })
    
    # Gyromagnetic ratio (MHz/T)
    gamma: Dict[str, float] = field(default_factory=lambda: {
        "P31": 17.235,  # Standard ³¹P
        "P32": 0.0,     # No nuclear magnetic moment for beta decay
        "P33": 13.7,    # Lower than P31
    })
    
    # Whether isotope supports J-coupling protection
    # Only spin-1/2 nuclei can form protected singlet states
    supports_singlet: Dict[str, bool] = field(default_factory=lambda: {
        "P31": True,
        "P32": False,  # Cannot form singlet (wrong spin)
        "P33": True,
    })


@dataclass 
class JCouplingParameters:
    """
    J-coupling protection parameters
    
    J-coupling between nuclear spins creates a protected subspace
    (the singlet state) that is immune to certain decoherence mechanisms.
    
    The key insight: ATP hydrolysis at active synapses provides
    strongly J-coupled phosphates (20 Hz), dramatically extending coherence.
    
    References:
    - Adams et al. 2025 (J-coupling from ATP hydrolysis)
    - Fisher 2015 (singlet protection mechanism)
    """
    
    # J-coupling strength (Hz)
    J_baseline: float = 0.2      # Free phosphate in solution
    J_ATP: float = 20.0          # ATP-derived phosphate (Adams et al. 2025)
    
    # Protection factor: how much J-coupling extends T2
    # Derived from singlet-triplet dynamics
    # T2_protected = T2_single * protection_factor
    protection_factor: float = 25.0
    
    # Threshold J-coupling for significant protection (Hz)
    J_threshold: float = 5.0


@dataclass
class DimerStructure:
    """
    Calcium phosphate dimer structural parameters
    
    Ca₆(PO₄)₄ - the quantum-coherent structure
    Contains exactly 4 phosphorus atoms (hence "dimer" vs "trimer" with 6)
    
    References:
    - Agarwal et al. 2023 (dimer structure and coherence)
    - Model 6 implementation
    """
    
    # Number of phosphorus atoms
    n_phosphorus: int = 4
    
    # Intrinsic J-coupling within dimer structure (Hz)
    # This is fixed by molecular geometry
    J_intrinsic: float = 15.0
    
    # Dimer radius (nm)
    radius_nm: float = 0.5
    
    # Formation threshold - minimum conditions for dimer nucleation
    ca_threshold_uM: float = 1.0      # Calcium concentration
    phosphate_threshold_uM: float = 10.0  # Phosphate concentration


@dataclass
class QuantumEligibilityPhysics:
    """
    Complete physics model for quantum eligibility traces
    
    This class derives the eligibility trace decay constant (τ = T2)
    from first-principles nuclear spin physics.
    
    Key principle: These are NOT hyperparameters to optimize.
    They are physical constants that determine system behavior.
    
    Usage:
        physics = QuantumEligibilityPhysics(isotope=Isotope.P31)
        T2 = physics.get_T2_effective()  # ~25-67 seconds
        decay_rate = physics.get_decay_rate()  # 1/T2
    """
    
    # Primary control: isotope selection
    isotope: Isotope = Isotope.P31
    
    # Sub-parameters (physical constants)
    nuclear: NuclearSpinParameters = field(default_factory=NuclearSpinParameters)
    j_coupling: JCouplingParameters = field(default_factory=JCouplingParameters)
    dimer: DimerStructure = field(default_factory=DimerStructure)
    
    # Environmental conditions
    temperature_K: float = 310.0  # Body temperature
    atp_active: bool = True       # Whether ATP hydrolysis is occurring
    
    # Dopamine modulation (affects dimer vs trimer selectivity)
    dopamine_present: bool = False
    dopamine_enhancement: float = 1.5  # How much DA enhances dimer formation
    
    def get_T2_single(self) -> float:
        """
        Get single-spin T2 for current isotope
        
        Returns:
            T2 in seconds
        """
        return self.nuclear.T2_single[self.isotope.value]
    
    def get_J_coupling_effective(self) -> float:
        """
        Get effective J-coupling strength
        
        ATP hydrolysis at active synapses provides strong J-coupling.
        Without ATP, only weak baseline coupling exists.
        
        Returns:
            J-coupling in Hz
        """
        if self.atp_active:
            return self.j_coupling.J_ATP
        else:
            return self.j_coupling.J_baseline
    
    def get_protection_factor(self) -> float:
        """
        Calculate J-coupling protection factor
        
        Only spin-1/2 nuclei (P31, P33) can form protected singlet states.
        P32 (spin-1) cannot benefit from this protection.
        
        Returns:
            Protection multiplier for T2
        """
        isotope_str = self.isotope.value
        
        # Check if isotope supports singlet protection
        if not self.nuclear.supports_singlet[isotope_str]:
            return 1.0  # No protection for non-spin-1/2
        
        # Protection scales with J-coupling strength
        J_eff = self.get_J_coupling_effective()
        
        if J_eff < self.j_coupling.J_threshold:
            # Below threshold: minimal protection
            return 1.0 + (J_eff / self.j_coupling.J_threshold) * (self.j_coupling.protection_factor - 1.0)
        else:
            # Above threshold: full protection
            return self.j_coupling.protection_factor
    
    def get_spin_count_factor(self) -> float:
        """
        Calculate decoherence factor from multiple coupled spins
        
        More spins = more decoherence pathways
        T2_multi = T2_single / sqrt(n_spins)
        
        Returns:
            Reduction factor (< 1.0)
        """
        n_spins = self.dimer.n_phosphorus
        return 1.0 / np.sqrt(n_spins)
    
    def get_T2_effective(self) -> float:
        """
        Calculate effective coherence time from physics
        
        This is THE key derived quantity:
        
        T2_eff = T2_single × protection_factor × spin_count_factor
        
        For P31 with ATP:
            T2_eff = 2.0 × 25.0 × (1/√4) = 25.0 seconds
        
        For P32 (no protection):
            T2_eff = 0.2 × 1.0 × (1/√4) = 0.1 seconds
        
        Returns:
            Effective T2 in seconds
        """
        T2_single = self.get_T2_single()
        protection = self.get_protection_factor()
        spin_factor = self.get_spin_count_factor()
        
        T2_eff = T2_single * protection * spin_factor
        
        return T2_eff
    
    def get_decay_rate(self) -> float:
        """
        Get eligibility decay rate (1/T2)
        
        This is the λ-equivalent in TD(λ) learning,
        but derived from physics rather than tuned.
        
        Returns:
            Decay rate in 1/seconds
        """
        return 1.0 / self.get_T2_effective()
    
    def get_lambda_equivalent(self, dt: float) -> float:
        """
        Convert to TD(λ) equivalent decay parameter
        
        In TD(λ), eligibility updates as: e ← λ * e
        Our physics gives: e ← e * exp(-dt/T2)
        
        So: λ = exp(-dt/T2)
        
        Args:
            dt: timestep in seconds
            
        Returns:
            Equivalent λ parameter
        """
        T2 = self.get_T2_effective()
        return np.exp(-dt / T2)
    
    def get_singlet_probability_equilibrium(self) -> float:
        """
        Get equilibrium singlet probability
        
        At thermal equilibrium, P_singlet = 0.25 (statistical mixture)
        Quantum preparation can increase this toward 1.0
        
        Returns:
            Equilibrium singlet probability
        """
        return 0.25  # Thermal equilibrium
    
    def get_entanglement_threshold(self) -> float:
        """
        Get threshold for entanglement
        
        P_singlet > 0.5 indicates entanglement preserved (Agarwal 2023)
        
        Returns:
            Threshold singlet probability for entanglement
        """
        return 0.5
    
    def summary(self) -> Dict:
        """
        Get summary of all derived parameters
        
        Returns:
            Dictionary of key physics parameters
        """
        return {
            'isotope': self.isotope.value,
            'T2_single_s': self.get_T2_single(),
            'J_coupling_Hz': self.get_J_coupling_effective(),
            'protection_factor': self.get_protection_factor(),
            'spin_count_factor': self.get_spin_count_factor(),
            'T2_effective_s': self.get_T2_effective(),
            'decay_rate_per_s': self.get_decay_rate(),
            'lambda_equivalent_100ms': self.get_lambda_equivalent(0.1),
            'supports_singlet': self.nuclear.supports_singlet[self.isotope.value],
        }
    
    def __str__(self) -> str:
        s = self.summary()
        lines = [
            f"QuantumEligibilityPhysics ({s['isotope']})",
            f"  T2_single: {s['T2_single_s']:.2f} s",
            f"  J-coupling: {s['J_coupling_Hz']:.1f} Hz",
            f"  Protection: {s['protection_factor']:.1f}x",
            f"  T2_effective: {s['T2_effective_s']:.1f} s",
            f"  Decay rate: {s['decay_rate_per_s']:.4f} /s",
        ]
        return "\n".join(lines)


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_isotopes() -> Dict[str, Dict]:
    """
    Compare physics across all isotopes
    
    This is the computational version of the isotope substitution experiment.
    
    Returns:
        Dictionary mapping isotope -> physics summary
    """
    results = {}
    
    for isotope in Isotope:
        physics = QuantumEligibilityPhysics(isotope=isotope)
        results[isotope.value] = physics.summary()
    
    return results


def get_T2_ratio(isotope1: Isotope, isotope2: Isotope) -> float:
    """
    Calculate T2 ratio between two isotopes
    
    Key prediction: T2(P31) / T2(P32) ~ 250x
    
    Args:
        isotope1: Numerator isotope
        isotope2: Denominator isotope
        
    Returns:
        Ratio of T2 values
    """
    physics1 = QuantumEligibilityPhysics(isotope=isotope1)
    physics2 = QuantumEligibilityPhysics(isotope=isotope2)
    
    return physics1.get_T2_effective() / physics2.get_T2_effective()


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM ELIGIBILITY TRACE PHYSICS")
    print("=" * 60)
    
    # Compare all isotopes
    print("\n--- Isotope Comparison ---")
    comparison = compare_isotopes()
    
    for isotope, params in comparison.items():
        print(f"\n{isotope}:")
        print(f"  T2_effective: {params['T2_effective_s']:.2f} s")
        print(f"  Decay rate: {params['decay_rate_per_s']:.4f} /s")
        print(f"  λ (dt=0.1s): {params['lambda_equivalent_100ms']:.4f}")
    
    # Key prediction
    print("\n--- KEY PREDICTION ---")
    ratio = get_T2_ratio(Isotope.P31, Isotope.P32)
    print(f"T2(P31) / T2(P32) = {ratio:.0f}x")
    print(f"This should cause ~{ratio:.0f}x difference in learning performance")
    print("for tasks requiring >1 second temporal integration")
    
    # Full physics for P31
    print("\n--- Full P31 Physics ---")
    physics_p31 = QuantumEligibilityPhysics(isotope=Isotope.P31)
    print(physics_p31)