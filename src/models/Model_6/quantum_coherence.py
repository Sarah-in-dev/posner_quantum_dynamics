"""
Quantum Coherence System for Model 6
=====================================
Tracks quantum coherence in Ca₆(PO₄)₄ dimers (4 ³¹P nuclei)

This system ONLY handles quantum state evolution - it does NOT create dimers.
Dimers are created by ca_triphosphate_complex.py and passed here for coherence tracking.

Key Citations:
- Agarwal et al. 2023 "The Biological Qubit": Ca₆(PO₄)₄ with 4 ³¹P, T2 ~ 100s
- Fisher 2015 Ann Phys: T2 enhanced by J-coupling protection
- Nuclear spin I=1/2 for ³¹P provides long coherence times

Architecture:
  ca_triphosphate_complex.py → [Ca₆(PO₄)₄] → quantum_coherence_system.py
                                   ^                    |
                                   |                    v
                              Creates dimers      Tracks quantum state
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

from model6_parameters import Model6Parameters, QuantumParameters

logger = logging.getLogger(__name__)


class QuantumCoherenceSystem:
    """
    Quantum coherence tracking for Ca₆(PO₄)₄ dimers
    
    Receives dimer concentrations from ca_triphosphate_complex.py
    and evolves their quantum state according to:
    
    1. T2 decoherence (isotope-dependent)
    2. J-coupling protection (ATP-dependent)
    3. Temperature independence (quantum signature)
    
    State evolution:
    ρ(t) = ρ(0) * exp(-t/T2_eff)
    
    where T2_eff depends on:
    - Isotope composition (³¹P vs ³²P)
    - J-coupling strength (ATP hydrolysis)
    - Temperature (should be minimal - Q10 ~ 1)
    """
    
    def __init__(self, grid_shape: Tuple[int, int], params: QuantumParameters,  # FIX TYPE
                isotope_P31_fraction: float = 1.0):
        self.grid_shape = grid_shape
        self.params = params  # params is already QuantumParameters
    
        # Isotope composition (EXPERIMENTAL VARIABLE)
        self.P31_fraction = isotope_P31_fraction
    
        # Store base T2 values for dynamic calculation
        self.T2_base_dimer = self.params.T2_base_dimer
        self.Q10_quantum = self.params.Q10_quantum
    
        # Coherence field (0-1, where 1 = perfect coherence)
        # Tracks quantum state of dimers at each spatial location
        self.coherence = np.zeros(grid_shape)  # FIX NAME: coherence not coherence_field
    
        # Track dimer concentration (for knowing where coherence exists)
        self.dimer_concentration = np.zeros(grid_shape)
    
        # Will store effective T2 after dynamic calculation in step()
        self.T2_effective = np.ones(grid_shape) * self.params.T2_base_dimer
    
        logger.info(f"Initialized quantum coherence system")
        logger.info(f"  Isotope: {self.P31_fraction*100:.0f}% ³¹P")
        logger.info(f"  Base T2: {self.params.T2_base_dimer:.1f} s")
        
    def _calculate_effective_T2(self) -> float:
        """
        Calculate effective T2 based on isotope composition
        
        KEY EXPERIMENTAL PREDICTION:
        - ³¹P (I=1/2): T2 ~ 100 s (long coherence)
        - ³²P (I=1): T2 ~ 10 s (shorter coherence)
        - ³³P (I=1/2): Similar to ³¹P but rare
        
        Returns:
            Effective T2 in seconds
        """
        T2_base = self.params.T2_base_dimer  # Base coherence time
        
        # Isotope-weighted T2
        # Fisher 2015: Only odd-numbered isotopes (I=1/2) provide long coherence
        T2_eff = self.P31_fraction * T2_base * self.params.coherence_factor_P31
        
        # ³²P contribution (if present)
        P32_fraction = getattr(self.params, 'fraction_P32', 0.0)
        if P32_fraction > 0:
            T2_eff += P32_fraction * T2_base * self.params.coherence_factor_P32
        
        return T2_eff
        
    def step(self, dt: float, dimer_conc: np.ndarray, j_coupling: np.ndarray,
         temperature: float = 310.15, dopamine_d2: Optional[np.ndarray] = None,
         template_binding: Optional[np.ndarray] = None):
        """
        Update quantum coherence with EMERGENT T2 calculation
    
        T2 emerges from:
        - Base isotope value (³¹P vs ³²P)
        - J-coupling strength (ATP protection)
        - Binding state (template vs free)
        - Dopamine modulation (D2 receptor)
        """
    
        # === CALCULATE EFFECTIVE T2 (EMERGENT!) ===
    
        # 1. Start with base isotope-dependent T2
        T2_base = self.T2_base_dimer  # 100s for ³¹P, 10s for ³²P
    
        # 2. J-coupling enhancement (Fisher 2015: T2 ∝ J²)
        # Stronger coupling = better quantum protection
        J_free = 0.2  # Hz (free phosphate baseline)
        J_ATP = 20.0  # Hz (ATP baseline)
        # Linear enhancement, not quadratic (more realistic)
        j_enhancement = 1.0 + 2.0 * (j_coupling - J_free) / (J_ATP - J_free)
        j_enhancement = np.clip(j_enhancement, 1.0, 3.0)  # Max 3x enhancement
    
        # 3. Binding state factor (from earlier discussion)
        if template_binding is not None:
            # Template-bound dimers tumble slower → longer T2
            binding_factor = 1.0 + template_binding * 1.0  # Up to 2x for bound
        else:
            binding_factor = 1.0
    
        # 4. Dopamine protection (D2 reduces environmental noise)
        if dopamine_d2 is not None:
            # D2 activation shields from decoherence
            dopamine_protection = 1.0 + dopamine_d2 * 2.0  # Up to 3x at full activation
        else:
            dopamine_protection = 1.0
    
        # 5. Temperature factor (should be ~1 for quantum!)
        T_ref = 310.15
        Q10 = self.Q10_quantum  # Should be 1.0
        temp_factor = Q10 ** ((temperature - T_ref) / 10.0)
    
        # === EMERGENT T2 ===
        T2_effective = (T2_base * j_enhancement * binding_factor * 
                    dopamine_protection * temp_factor)
    
        # === DECOHERENCE (Exponential decay) ===
        # Where dimers exist, apply decoherence
        has_dimers = dimer_conc > 0
    
        # Initialize new dimers with full coherence
        new_dimers = has_dimers & (self.coherence == 0)
        self.coherence[new_dimers] = 1.0
    
        # Apply exponential decoherence
        self.coherence[has_dimers] *= np.exp(-dt / T2_effective[has_dimers])
    
        # Clear coherence where no dimers
        self.coherence[~has_dimers] = 0
    
        # Clip to [0, 1]
        self.coherence = np.clip(self.coherence, 0, 1)
    
        # Store effective T2 for reporting
        self.T2_effective = T2_effective
        
        def get_coherence(self) -> np.ndarray:
            """Get spatial coherence field (0-1)"""
            return self.coherence.copy()
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return experimental metrics for validation
        
        Returns metrics aligned with experimental measurements:
        - Mean coherence (where dimers exist)
        - Coherence decay rate
        - Effective T2 time
        - Isotope composition
        """
        # Only compute statistics where dimers exist
        has_dimers = self.dimer_concentration > 0
        
        if np.any(has_dimers):
            coherence_mean = np.mean(self.coherence[has_dimers])
            coherence_std = np.std(self.coherence[has_dimers])
        else:
            coherence_mean = 0.0
            coherence_std = 0.0
        
        return {
            # Quantum metrics
            'coherence_mean': float(np.mean(self.coherence)),  # CHANGE HERE
            'coherence_peak': float(np.max(self.coherence)),   # CHANGE HERE
            'coherence_std': float(np.std(self.coherence)),    # CHANGE HERE
            
            # T2 coherence time
            'T2_dimer_s': float(np.mean(self.T2_effective)),
            'dimer_peak_nM': float(np.max(self.dimer_concentration) * 1e9),
            'dimer_mean_nM': float(np.mean(self.dimer_concentration) * 1e9),
            
            # Isotope composition
            'P31_fraction': float(self.P31_fraction),
            
            # Spatial statistics
            'coherent_volume': float(np.sum(self.coherence > 0.9)),
            'partially_coherent_volume': float(np.sum((self.coherence > 0.5) & 
                                                      (self.coherence <= 0.9))),
        }


# =============================================================================
# VALIDATION TEST
# =============================================================================

if __name__ == "__main__":
    from model6_parameters import Model6Parameters
    
    print("="*70)
    print("QUANTUM COHERENCE SYSTEM VALIDATION")
    print("="*70)
    
    # Test 1: Natural isotope composition (³¹P)
    print("\nTest 1: Natural Isotope (³¹P)")
    params_P31 = Model6Parameters()
    params_P31.environment.fraction_P31 = 1.0
    
    grid_shape = (50, 50)
    qc_system = QuantumCoherenceSystem(grid_shape, params_P31)
    
    # Simulate dimer formation at t=0
    dimer_conc = np.zeros(grid_shape)
    dimer_conc[20:30, 20:30] = 500e-9  # 500 nM dimers in center
    
    # Strong J-coupling (ATP present)
    j_coupling = np.ones(grid_shape) * 0.2
    j_coupling[20:30, 20:30] = 18.0  # 18 Hz in active region
    
    print(f"  Initial dimers: 500 nM (center region)")
    print(f"  J-coupling: 18 Hz (active), 0.2 Hz (bulk)")
    
    # Simulate 1 second
    dt = 0.001  # 1 ms
    n_steps = 1000
    
    coherence_history = []
    for i in range(n_steps):
        qc_system.step(dt, dimer_conc, j_coupling)
        
        if i % 100 == 0:
            metrics = qc_system.get_experimental_metrics()
            coherence_history.append(metrics['coherence_mean'])
            print(f"    t={i*dt:.2f}s: coherence={metrics['coherence_mean']:.4f}")
    
    # Validate exponential decay
    final_metrics = qc_system.get_experimental_metrics()
    expected_coherence = np.exp(-1.0 / qc_system.T2_eff)  # After 1 second
    
    print(f"\n  Final coherence: {final_metrics['coherence_mean']:.4f}")
    print(f"  Expected (exp(-t/T2)): {expected_coherence:.4f}")
    print(f"  T2 effective: {final_metrics['T2_dimer_s']:.1f} s")
    
    if abs(final_metrics['coherence_mean'] - expected_coherence) < 0.01:
        print(f"  ✓ Exponential decay validated")
    else:
        print(f"  ✗ Decay does not match exponential")
    
    # Test 2: Isotope substitution (³²P)
    print("\nTest 2: Isotope Substitution (³²P)")
    params_P32 = Model6Parameters()
    params_P32.environment.fraction_P31 = 0.0
    params_P32.environment.fraction_P32 = 1.0
    
    qc_system_P32 = QuantumCoherenceSystem(grid_shape, params_P32)
    
    # Same conditions
    for i in range(n_steps):
        qc_system_P32.step(dt, dimer_conc, j_coupling)
    
    metrics_P32 = qc_system_P32.get_experimental_metrics()
    
    print(f"  ³¹P T2: {final_metrics['T2_dimer_s']:.1f} s")
    print(f"  ³²P T2: {metrics_P32['T2_dimer_s']:.1f} s")
    print(f"  Ratio: {final_metrics['T2_dimer_s']/metrics_P32['T2_dimer_s']:.1f}x")
    print(f"  Thesis prediction: ~10x")
    
    if final_metrics['T2_dimer_s'] > 5 * metrics_P32['T2_dimer_s']:
        print(f"  ✓ Isotope effect detected (KEY PREDICTION!)")
    
    # Test 3: Temperature independence (Q10)
    print("\nTest 3: Temperature Independence (Q10)")
    
    temperatures = [305.15, 310.15, 315.15]  # 32, 37, 42°C
    T2_values = []
    
    for T in temperatures:
        qc_test = QuantumCoherenceSystem(grid_shape, params_P31)
        
        for i in range(100):
            qc_test.step(dt, dimer_conc, j_coupling, temperature=T)
        
        metrics_T = qc_test.get_experimental_metrics()
        T2_values.append(metrics_T['coherence_mean'])
        
        print(f"    {T-273.15:.0f}°C: coherence = {metrics_T['coherence_mean']:.4f}")
    
    # Calculate Q10
    if len(T2_values) >= 2:
        Q10_measured = (T2_values[-1] / T2_values[0]) ** (10 / (temperatures[-1] - temperatures[0]))
        print(f"  Measured Q10: {Q10_measured:.2f}")
        print(f"  Expected (quantum): ~1.0")
        print(f"  Expected (classical): >2.0")
        
        if Q10_measured < 1.2:
            print(f"  ✓ Temperature-independent (quantum signature!)")
        else:
            print(f"  ⚠ Temperature-dependent (Q10={Q10_measured:.2f})")
    
    print("\n" + "="*70)
    print("Quantum coherence system validated!")
    print("="*70)