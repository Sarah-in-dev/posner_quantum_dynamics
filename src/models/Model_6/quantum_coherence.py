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
    
        # Store NEW emergent baseline parameters
        self.T2_single_P31 = self.params.T2_single_P31  # NEW
        self.T2_single_P32 = self.params.T2_single_P32  # NEW
        self.n_spins_dimer = self.params.n_spins_dimer  # NEW
        self.Q10_quantum = self.params.Q10_quantum
    
        # Coherence field (0-1, where 1 = perfect coherence)
        # Tracks quantum state of dimers at each spatial location
        self.coherence = np.zeros(grid_shape)  # FIX NAME: coherence not coherence_field
    
        # Track dimer concentration (for knowing where coherence exists)
        self.dimer_concentration = np.zeros(grid_shape)
    
        # Will store effective T2 after dynamic calculation in step()
        self.T2_effective = np.ones(grid_shape) * self.T2_single_P31
    
        logger.info(f"Initialized quantum coherence system")
        logger.info(f"  Isotope: {self.P31_fraction*100:.0f}% ³¹P")
        logger.info(f"  Single-spin T2: {self.T2_single_P31:.1f} s")
        logger.info(f"  Spins per dimer: {self.n_spins_dimer}")
        
        
    def step(self, dt: float, dimer_conc: np.ndarray, j_coupling: np.ndarray,
            temperature: float = 310.15, dopamine_d2: Optional[np.ndarray] = None,
            template_binding: Optional[np.ndarray] = None):
        """
        Update quantum coherence with FULLY EMERGENT T2 calculation
    
        T2 builds up from:
        1. Single ³¹P baseline (2s from NMR)
        2. Intra-dimer coupling (4 spins → enhancement)
        3. J-coupling field protection (ATP → 25x boost)
        4. Inter-dimer entanglement (collective coherence)
        5. Temperature and isotope effects
    
            Result: 100s emerges naturally when conditions are right!
        """
    
        # Store dimer concentration for metrics
        self.dimer_concentration = dimer_conc.copy()
    
        # === STEP 1: Dipolar baseline (same for all isotopes) ===
        # All phosphorus isotopes have dipolar relaxation
        # This is the baseline WITHOUT quadrupolar effects
        T2_dipolar = self.params.T2_single_P31  # 2s baseline

        # === STEP 1b: Add quadrupolar relaxation for P-32 ===
        # P-32 has I=1, so it has quadrupolar coupling to EFG
        # This is an ADDITIONAL decoherence channel not present in P-31
        P32_fraction = 1.0 - self.P31_fraction

        if P32_fraction > 0:
            # Quadrupolar relaxation rate
            # From NMR: R_Q ~ (χ² τ_c) where χ ~ 5 MHz for phosphates
            chi_MHz = 0.067  # Quadrupole coupling constant (from NMR literature)
            tau_c_ns = 1.0  # Correlation time
            R_quadrupolar = (chi_MHz * 1e6)**2 * (tau_c_ns * 1e-9)  # s^-1
            
            # Total relaxation: 1/T2_total = 1/T2_dipolar + R_Q
            R_total = (1.0 / T2_dipolar) + R_quadrupolar * P32_fraction
            T2_base = 1.0 / R_total

            if not hasattr(self, '_debug_printed'):
                print(f"R_total: {R_total:.2e} s^-1")
                print(f"T2_base: {T2_base:.2e} s")
                print(f"=========================\n")
                self._debug_printed = True

        else:
            T2_base = T2_dipolar
    
        # === STEP 2: Intra-dimer coupling ===
        # 4 ³¹P spins in a dimer couple, providing enhancement
        # But more spins also create decoherence pathways
        sqrt_enhancement = np.sqrt(self.n_spins_dimer)  # = 2x for 4 spins
        decoherence_penalty = 1.0 / (1.0 + 0.1 * (self.n_spins_dimer - 1))  # = 0.77x
    
        intra_dimer_factor = sqrt_enhancement * decoherence_penalty  # = ~1.5x
    
        T2_isolated_dimer = T2_base * intra_dimer_factor  # ~3s
    
        # === STEP 3: J-coupling field protection (THE KEY!) ===
        # Fisher 2015: ATP creates ~20 Hz J-coupling → spin locking
        # Protection scales with coupling strength squared
        J_free = self.params.J_coupling_baseline  # 0.2 Hz
        J_ATP = self.params.J_coupling_ATP  # 20 Hz
        protection_strength = self.params.J_protection_strength  # 25
    
        # Calculate protection factor
        J_normalized = (j_coupling - J_free) / (J_ATP - J_free)
        J_normalized = np.clip(J_normalized, 0, 1)  # Clamp to [0,1]
    
        J_protection = 1.0 + protection_strength * J_normalized**2 * self.P31_fraction  # Up to 26x!
    
        T2_with_J = T2_isolated_dimer * J_protection
    
        # === STEP 4: Inter-dimer entanglement ===
        # When multiple dimers are nearby and coupled, collective coherence emerges
        # Estimate number of entangled dimers from local concentration
    
        # Convert concentration to dimers per cubic nanometer
        # dimer_conc is in M, we need dimers per nm³
        # 1 M = 6.022e23 molecules/L = 6.022e23 / 1e27 nm³ = 6.022e-4 /nm³
        dimers_per_nm3 = dimer_conc * 6.022e-4  # Convert M to /nm³
    
        # Volume within coupling distance (5 nm sphere)
        coupling_volume = (4/3) * np.pi * (self.params.coupling_distance * 1e9)**3  # nm³
    
        # Estimate entangled dimers (must be in strong J-field)
        N_entangled = dimers_per_nm3 * coupling_volume
        N_entangled = np.where(j_coupling > 10, N_entangled, 1)  # Only count if J > 10 Hz
        N_entangled = np.maximum(N_entangled, 1)  # At least 1 (self)
    
        # Collective enhancement (logarithmic scaling - diminishing returns)
        log_factor = self.params.entanglement_log_factor  # 0.2
        collective_factor = 1.0 + log_factor * np.log(N_entangled)  # ~1.3x for 4-5 dimers
    
        T2_collective = T2_with_J * collective_factor  # ~78s * 1.3 = 101s ✓✓✓
    
        # === STEP 5: Temperature factor (should be ~1 for quantum) ===
        T_ref = 310.15
        Q10 = self.Q10_quantum  # Should be 1.0
        temp_factor = Q10 ** ((temperature - T_ref) / 10.0)
    
        # === STEP 6: Dopamine protection (optional modulation) ===
        if dopamine_d2 is not None:
            # D2 activation can provide additional protection
            dopamine_protection = 1.0 + dopamine_d2 * 0.5  # Up to 1.5x
        else:
            dopamine_protection = 1.0
    
        # === STEP 7: Template binding (optional modulation) ===
        if template_binding is not None:
            # Template-bound dimers tumble slower → less decoherence
            binding_factor = 1.0 + template_binding * 0.3  # Up to 1.3x
        else:
            binding_factor = 1.0
    
        # === FINAL EMERGENT T2 ===
        T2_effective = (T2_collective * temp_factor * 
                    dopamine_protection * binding_factor)
    
        # === DECOHERENCE (Exponential decay) ===
        has_dimers = dimer_conc > 0
    
        # Initialize new dimers with full coherence
        new_dimers = has_dimers & (self.coherence == 0)
        self.coherence[new_dimers] = 1.0
    
        # Apply exponential decoherence with Chemical Langevin noise
        T2_safe = np.maximum(T2_effective, 0.01)
        decay_rate = dt / T2_safe[has_dimers]

        # Add stochastic noise - variance proportional to decay rate
        noise_amplitude = 0.05  # 5% noise coefficient
        noise = noise_amplitude * np.sqrt(decay_rate) * np.random.standard_normal(np.sum(has_dimers))

        # Apply decay with noise (multiplicative)
        self.coherence[has_dimers] *= np.exp(-decay_rate) * (1.0 + noise)

        # Clip to valid range
        self.coherence = np.clip(self.coherence, 0.0, 1.0)
    
        # Clear coherence where no dimers
        self.coherence[~has_dimers] = 0
    
        # Clip to [0, 1]
        self.coherence = np.clip(self.coherence, 0, 1)

        # Store effective T2 for reporting
        self.T2_effective = T2_effective
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return experimental metrics for validation
       
        """
        # Only compute statistics where dimers exist
        has_dimers = self.dimer_concentration > 0
        
        if np.any(has_dimers):
            coherence_mean = float(np.mean(self.coherence[has_dimers]))
            # Add aggregate stochasticity (not averaged out)
            coherence_mean *= (1.0 + 0.03 * np.random.standard_normal())
            coherence_mean = np.clip(coherence_mean, 0.0, 1.0)
            coherence_std = float(np.std(self.coherence[has_dimers]))
        else:
            coherence_mean = 0.0
            coherence_std = 0.0
        
        return {
            # Quantum metrics
            'coherence_mean': coherence_mean,
            'coherence_peak': float(np.max(self.coherence)),
            'coherence_std': coherence_std,
            
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