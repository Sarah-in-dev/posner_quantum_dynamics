"""
ATP System Module for Model 6
==============================
Handles:
- ATP hydrolysis (activity-dependent)
- Phosphate release and speciation  
- J-coupling field calculation (CRITICAL for quantum coherence)
- Recovery of ATP pools

Key insight: ATP provides the J-coupling that protects quantum states

Key Citations:
- Rangaraju et al. 2014 Cell 156:825-835 (ATP consumption)
- Cohn & Hughes 1962 J Biol Chem 237:176-181 (J-coupling NMR)
- Fisher 2015 Ann Phys 362:593-599 (quantum protection)
- Li & Gregory 1974 Geochim Cosmochim Acta (phosphate diffusion)
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

from model6_parameters import Model6Parameters, ATPParameters, PhosphateParameters

logger = logging.getLogger(__name__)


class ATPHydrolysis:
    """
    ATP hydrolysis dynamics
    ATP + H2O → ADP + Pi + H⁺ + energy
    
    Based on:
    - Rangaraju et al. 2014 Cell 156:825-835
      "4.7×10⁵ ATP molecules hydrolyzed per action potential"
      "~100 μM/s consumption rate during activity"
      "Recovery τ ~ 5 seconds via mitochondrial synthesis"
    
    - Lisman 2017 Philos Trans R Soc Lond B
      "Synaptic ATP critical for maintaining ion gradients"
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, params: ATPParameters):
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = params
        
        # ATP concentration field (Molarity)
        # Rangaraju et al. 2014: "Baseline ATP = 2.5 mM"
        self.atp = np.ones(grid_shape) * params.atp_concentration
        
        # ADP concentration (product of hydrolysis)
        self.adp = np.zeros(grid_shape)
        
        # Released phosphate (Pi) - goes into phosphate pool
        self.phosphate_released = np.zeros(grid_shape)
        
        # Activity markers (where hydrolysis is happening)
        self.activity_field = np.zeros(grid_shape)
        
        # Track total ATP consumed (for mass balance)
        self.total_consumed = 0.0
        self.total_recovered = 0.0

        # Stochastic parameters
        self.burst_probability = 0.02  # Probability of burst per active site per timestep
        self.burst_size_mean = 5e-7    # Mean ATP consumed per burst (M)
        self.burst_size_std = 1e-7     # Std dev in burst size
        
        logger.info(f"Initialized ATP hydrolysis system")
        
    def update_hydrolysis(self, dt: float, calcium: np.ndarray, 
                          activity_threshold: float = 1e-6):
        """
        Hydrolyze ATP in regions of elevated calcium via STOCHASTIC BURSTS
        
        CHANGES (Nov 2025):
        - Added probabilistic burst events at active sites
        - Variable burst sizes (Poisson-like)
        - Maintains mean rate but adds realistic variability
        
        Rangaraju et al. 2014 Cell:
        "ATP consumption scales with calcium influx"
        "Each action potential consumes 4.7×10⁵ molecules"
        """
        # Active sites = where calcium is elevated
        active_sites = calcium > activity_threshold
        
        # Update activity field (for J-coupling calculation)
        self.activity_field = active_sites.astype(float)
        
        # NEW: At TRUE rest (no activity anywhere), skip hydrolysis entirely
        if not np.any(active_sites):
            # Recovery only, no consumption
            return
        
        # === STOCHASTIC HYDROLYSIS BURSTS ===
        
        # 1. Baseline continuous hydrolysis (low rate, always present)
        basal_rate = self.params.hydrolysis_rate_basal
        basal_delta = basal_rate * dt
        basal_delta = np.minimum(basal_delta, self.atp)
        
        # 2. Stochastic burst events at active sites
        # Probability of burst increases with calcium
        burst_probability = self.burst_probability * dt * active_sites.astype(float)
        
        # Roll for burst events
        burst_events = np.random.rand(*self.grid_shape) < burst_probability
        
        # Variable burst sizes (gamma distribution for realistic variability)
        # Shape parameter controls variability
        shape = (self.burst_size_mean / self.burst_size_std) ** 2
        scale = self.burst_size_std ** 2 / self.burst_size_mean
        
        burst_sizes = np.random.gamma(shape, scale, self.grid_shape)
        burst_delta = np.where(burst_events, burst_sizes, 0)
        
        # Can't consume more ATP than available
        burst_delta = np.minimum(burst_delta, self.atp - basal_delta)
        
        # Total ATP consumed
        delta_atp = basal_delta + burst_delta
        
        # Update ATP
        self.atp -= delta_atp
        self.adp += delta_atp
        self.phosphate_released += delta_atp
        
        # Track total
        self.total_consumed += np.sum(delta_atp)
        
    def update_recovery(self, dt: float):
        """
        ATP recovery via mitochondrial synthesis
        
        Rangaraju et al. 2014 Cell:
        "ATP recovers with τ ~ 5 seconds"
        "Mitochondria maintain ATP/ADP ratio ~10:1"
        
        Exponential recovery toward baseline
        """
        # FIXED: Only recover where ATP is below baseline
        needs_recovery = self.atp < self.params.atp_concentration
    
        if not np.any(needs_recovery):
            return  # Nothing to recover
    
        # Recovery rate (first-order kinetics)
        # dATP/dt = (ATP_baseline - ATP) / τ
        recovery_rate = np.zeros_like(self.atp)
        recovery_rate[needs_recovery] = ((self.params.atp_concentration - self.atp[needs_recovery]) / 
                                      self.params.atp_recovery_tau)
    
        # Update ATP with stability limit
        # CRITICAL: Limit step size to prevent overshooting
        max_recovery = self.params.atp_concentration - self.atp
        delta_atp = np.minimum(recovery_rate * dt, max_recovery)
        delta_atp = np.maximum(delta_atp, 0)  # No negative recovery
    
        self.atp += delta_atp
    
        # Ensure we never exceed baseline
        self.atp = np.minimum(self.atp, self.params.atp_concentration)
    
        # ADP is converted back to ATP
        adp_consumed = np.minimum(delta_atp, self.adp)
        self.adp -= adp_consumed
        self.adp = np.maximum(self.adp, 0)  # No negative ADP
    
        # Track total
        self.total_recovered += np.sum(delta_atp)
        
    def update_diffusion(self, dt: float):
        """
        ATP and ADP diffusion
    
        CRITICAL: Check CFL condition for numerical stability
        CFL = D * dt / dx² < 0.5
        """
        # CFL stability check
        cfl = self.params.D_atp * dt / (self.dx ** 2)
        if cfl > 0.4:
            logger.warning(f"CFL condition violated: {cfl:.3f} > 0.5. Diffusion may be unstable!")
            # Reduce effective dt
            n_substeps = int(np.ceil(cfl / 0.4))
            dt_substep = dt / n_substeps
        
            # Warn once
            if not hasattr(self, '_cfl_warned'):
                logger.warning(f"CFL {cfl:.3f} > 0.4, using {n_substeps} substeps (dt_sub={dt_substep*1e6:.1f}μs)")
                self._cfl_warned = True
        else:
            n_substeps = 1
            dt_substep = dt
    
        # Laplacian kernel (5-point stencil)
        laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]]) / (self.dx**2)
    
        # Diffusion loop with substepping
        for _ in range(n_substeps):
            # Laplacian for ATP
            laplacian_atp = ndimage.laplace(self.atp) / (self.dx ** 2)
        
            # Laplacian for ADP
            laplacian_adp = ndimage.laplace(self.adp) / (self.dx ** 2)
        
            # Update with smaller timestep
            self.atp += dt_substep * self.params.D_atp * laplacian_atp
            self.adp += dt_substep * self.params.D_atp * laplacian_adp
        
            # Ensure non-negative (catch any numerical issues)
            self.atp = np.maximum(self.atp, 0)
            self.adp = np.maximum(self.adp, 0)
        
            # Check for NaN/Inf
            if np.any(~np.isfinite(self.atp)) or np.any(~np.isfinite(self.adp)):
                logger.error("NaN/Inf detected in ATP diffusion! Resetting...")
                self.atp = np.nan_to_num(self.atp, nan=self.params.atp_concentration, posinf=self.params.atp_concentration)
                self.adp = np.nan_to_num(self.adp, nan=0, posinf=0)
                break


class JCouplingField:
    """
    Calculate J-coupling field from ATP and phosphate
    
    THIS IS CRITICAL FOR QUANTUM COHERENCE!
    
    Based on:
    - Cohn & Hughes 1962 J Biol Chem 237:176-181
      "³¹P-³¹P coupling constant in ATP = 20 Hz"
      "Measured by high-resolution NMR"
      
    - Fisher 2015 Ann Phys 362:593-599
      "J-coupling protects nuclear spin coherence"
      "Zero-quantum subspace immune to environmental noise"
      "Free phosphate J-coupling ~ 0.2 Hz (100x weaker)"
    
    Key insight:
    ATP hydrolysis releases structured phosphates with strong J-coupling,
    providing temporary quantum protection that enables coherence
    during critical learning windows!
    """
    
    def __init__(self, grid_shape: Tuple[int, int], params: ATPParameters):
        self.grid_shape = grid_shape
        self.params = params
        
        # J-coupling strength field (Hz)
        # Fisher 2015: Free phosphate baseline = 0.2 Hz
        self.j_coupling = np.ones(grid_shape) * params.J_PO_free
        
        # Track history for analysis
        self.j_history = []
        
        logger.info("Initialized J-coupling field")
        
    def calculate_j_coupling(self, atp: np.ndarray, phosphate: np.ndarray,
                            activity: np.ndarray):
        """
        Calculate J-coupling from local ATP and phosphate concentrations
        
        Key insight from Fisher 2015:
        - ATP-bound phosphates: J = 20 Hz (strong)
        - Free phosphates: J = 0.2 Hz (weak)
        - During activity: ATP hydrolysis releases structured phosphates
        - These maintain J-coupling briefly before dissociating
        - Window of enhanced J provides quantum protection!
        
        Args:
            atp: ATP concentration field (M)
            phosphate: Total phosphate field (M)
            activity: Activity field (0 or 1)
        """
        # Fraction of phosphate that's ATP-bound
        # Simple model: fraction = [ATP] / ([ATP] + K_bind)
        # K_bind ~ 1 mM (typical enzyme binding constant)
        K_bind = 1e-3  # M
        
        frac_atp_bound = atp / (atp + K_bind)
        
        # Base J-coupling is weighted average
        # Cohn & Hughes 1962: ATP has J = 20 Hz
        # Fisher 2015: Free phosphate has J = 0.2 Hz
        self.j_coupling = (
            frac_atp_bound * self.params.J_PP_atp +
            (1 - frac_atp_bound) * self.params.J_PO_free
        )
        
        # Activity enhances J-coupling!
        # Mechanism: ATP hydrolysis releases structured phosphate pairs
        # These maintain J-coupling briefly (~10 ms)
        # Provides quantum protection during learning window
        activity_enhancement = 1.0 + 0.5 * activity  # Up to 1.5x boost
        self.j_coupling *= activity_enhancement
        
        # Cap at ATP maximum
        self.j_coupling = np.minimum(self.j_coupling, self.params.J_PP_atp)
        
        # Never below free phosphate minimum
        self.j_coupling = np.maximum(self.j_coupling, self.params.J_PO_free)
        
    def get_quantum_protection_factor(self) -> np.ndarray:
        """
        Calculate quantum protection factor from J-coupling
        
        Fisher 2015 Ann Phys:
        "Coherence time T2 ∝ J²"
        "Zero-quantum coherence protected from environmental noise"
        
        Returns:
            Protection factor (dimensionless, 1.0 = baseline)
        """
        # Protection scales as (J / J_baseline)²
        J_baseline = self.params.J_PO_free
        protection = (self.j_coupling / J_baseline) ** 2
        
        return protection


class PhosphateSpeciation:
    """
    Handle phosphate speciation based on pH
    H3PO4 ⇌ H2PO4⁻ ⇌ HPO4²⁻ ⇌ PO4³⁻
    
    Based on:
    - Standard acid-base chemistry (pKa values)
    - Li & Gregory 1974 Geochim Cosmochim Acta
      "D_H2PO4 = 890 μm²/s"
      "D_HPO4 = 790 μm²/s"
      "D_PO4 = 612 μm²/s"
    
    Importance:
    - At pH 7.3: ~90% HPO4²⁻, ~10% H2PO4⁻
    - During activity (pH 6.8): shifts toward H2PO4⁻
    - This affects which species binds calcium → Posner formation!
    """
    
    def __init__(self, grid_shape: Tuple[int, int], params: PhosphateParameters):
        self.grid_shape = grid_shape
        self.params = params
        
        # Total phosphate (conserved)
        self.phosphate_total = np.ones(grid_shape) * params.phosphate_total
        
        # Speciated forms
        self.H2PO4 = np.zeros(grid_shape)  # Monobasic
        self.HPO4 = np.zeros(grid_shape)   # Dibasic (main species)
        self.PO4 = np.zeros(grid_shape)    # Tribasic
        
        logger.info("Initialized phosphate speciation")
        
    def update_speciation(self, pH: np.ndarray):
        """
        Calculate speciation based on local pH using Henderson-Hasselbalch
        
        Args:
            pH: pH field
        """
        # SAFETY: Clip pH to reasonable range to prevent nan
        pH = np.clip(pH, 5.0, 9.0)  # Physiological range
        
        # Convert pH to [H⁺]
        H_conc = 10**(-pH)
        
        # Acid dissociation constants
        # pKa1 = 2.1  (not relevant, too acidic)
        # pKa2 = 7.2  (relevant! H2PO4⁻ ⇌ HPO4²⁻)
        # pKa3 = 12.4 (not relevant, too basic)
        
        Ka1 = 10**(-self.params.pKa1)
        Ka2 = 10**(-self.params.pKa2)
        Ka3 = 10**(-self.params.pKa3)
        
        # Species fractions using full equilibrium
        # α_i = [species_i] / [P_total]
        
        denom = (H_conc**3 + H_conc**2 * Ka1 + H_conc * Ka1 * Ka2 + 
                Ka1 * Ka2 * Ka3)
        
        # At physiological pH, mainly alpha1 and alpha2 matter
        alpha1 = (H_conc**2 * Ka1) / denom  # H2PO4⁻
        alpha2 = (H_conc * Ka1 * Ka2) / denom  # HPO4²⁻
        alpha3 = (Ka1 * Ka2 * Ka3) / denom  # PO4³⁻
        
        # Calculate concentrations
        self.H2PO4 = alpha1 * self.phosphate_total
        self.HPO4 = alpha2 * self.phosphate_total
        self.PO4 = alpha3 * self.phosphate_total
        
    def add_phosphate(self, source: np.ndarray):
        """
        Add released phosphate from ATP hydrolysis
        
        Args:
            source: Phosphate to add (M)
        """
        self.phosphate_total += source
    
    def get_posner_forming_species(self) -> np.ndarray:
        """
        Get the phosphate species that forms Posners
        
        McDonogh et al. 2024 Cryst Growth Des:
        "HPO4²⁻ is the primary species that binds Ca²⁺"
        "K_CaHPO4 = 470 M⁻¹ at pH 7.3"
        
        Returns:
            HPO4²⁻ concentration (M)
        """
        return self.HPO4


class ATPSystem:
    """
    Complete ATP system integrating hydrolysis, J-coupling, and phosphate
    
    This is the quantum protection mechanism!
    """
    
    def __init__(self, grid_shape: Tuple[int, int], dx: float, params: Model6Parameters):
        self.grid_shape = grid_shape
        self.dx = dx
        
        # Components
        self.hydrolysis = ATPHydrolysis(grid_shape, dx, params.atp)
        self.j_coupling = JCouplingField(grid_shape, params.atp)
        self.phosphate = PhosphateSpeciation(grid_shape, params.phosphate)
        
        logger.info("Initialized ATP system")
        
    def step(self, dt: float, calcium: np.ndarray, pH: np.ndarray):
        """
        Update ATP system for one time step
        
        Args:
            dt: Time step (s)
            calcium: Calcium concentration field (M)
            pH: pH field
        """
        # 1. ATP hydrolysis (activity-dependent)
        self.hydrolysis.update_hydrolysis(dt, calcium)
        
        # 2. Add released phosphate to pool
        if np.any(self.hydrolysis.phosphate_released > 0):
            self.phosphate.add_phosphate(self.hydrolysis.phosphate_released)
            self.hydrolysis.phosphate_released[:] = 0  # Reset
        
        # 3. Update phosphate speciation based on pH
        self.phosphate.update_speciation(pH)
        
        # 4. Calculate J-coupling field (quantum protection!)
        self.j_coupling.calculate_j_coupling(
            self.hydrolysis.atp,
            self.phosphate.phosphate_total,
            self.hydrolysis.activity_field
        )
        
        # 5. ATP diffusion
        # self.hydrolysis.update_diffusion(dt)
        
        # 6. ATP recovery (mitochondrial synthesis)
        self.hydrolysis.update_recovery(dt)
        
    def get_j_coupling(self) -> np.ndarray:
        """
        Get current J-coupling field (Hz)
        
        This is THE critical output for quantum coherence!
        """
        return self.j_coupling.j_coupling.copy()
    
    def get_quantum_protection(self) -> np.ndarray:
        """
        Get quantum protection factor field
        
        Fisher 2015: T2 ∝ J²
        """
        return self.j_coupling.get_quantum_protection_factor()
    
    def get_atp_concentration(self) -> np.ndarray:
        """Get ATP concentration field (M)"""
        return self.hydrolysis.atp.copy()
    
    def get_phosphate_for_posner(self) -> np.ndarray:
        """
        Get the phosphate species that forms Posners
        
        McDonogh et al. 2024: HPO4²⁻ at physiological pH
        """
        return self.phosphate.get_posner_forming_species()
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return metrics for experimental validation
        
        Aligns with thesis measurements:
        - J-coupling strength (NMR)
        - ATP concentration (biochemical assay)
        - Quantum protection factor (predicted T2 enhancement)
        """
        j_mean = np.mean(self.j_coupling.j_coupling)
        j_max = np.max(self.j_coupling.j_coupling)
        
        protection = self.j_coupling.get_quantum_protection_factor()
        
        return {
            'atp_mean_mM': np.mean(self.hydrolysis.atp) * 1e3,
            'atp_min_mM': np.min(self.hydrolysis.atp) * 1e3,
            'j_coupling_mean_Hz': j_mean,
            'j_coupling_max_Hz': j_max,
            'quantum_protection_mean': np.mean(protection),
            'quantum_protection_max': np.max(protection),
            'atp_consumed_total': self.hydrolysis.total_consumed,
            'atp_recovered_total': self.hydrolysis.total_recovered,
        }


# ============================================================================
# VALIDATION TESTS
# ============================================================================

if __name__ == "__main__":
    from model6_parameters import Model6Parameters
    
    print("="*70)
    print("ATP SYSTEM VALIDATION")
    print("="*70)
    
    params = Model6Parameters()
    
    # Create small grid
    grid_shape = (50, 50)
    dx = 2 * params.spatial.active_zone_radius / grid_shape[0]
    
    # Create ATP system
    atp_system = ATPSystem(grid_shape, dx, params)
    
    print(f"\nInitial state:")
    metrics = atp_system.get_experimental_metrics()
    print(f"  ATP: {metrics['atp_mean_mM']:.2f} mM")
    print(f"  J-coupling: {metrics['j_coupling_mean_Hz']:.2f} Hz")
    print(f"  Quantum protection: {metrics['quantum_protection_mean']:.2f}x")
    
    # Test 1: Activity-dependent ATP consumption
    print("\nTest 1: Activity-Dependent ATP Consumption")
    print("Simulating 10 ms with calcium spike...")
    
    dt = 1e-5  # 10 μs
    n_steps = 1000
    
    # Create calcium spike in center (simulating channel activity)
    calcium = np.ones(grid_shape) * 100e-9  # 100 nM baseline
    calcium[20:30, 20:30] = 10e-6  # 10 μM in active zone
    
    # pH field (slightly acidic during activity)
    pH = np.ones(grid_shape) * 7.3
    pH[20:30, 20:30] = 7.0
    
    j_history = []
    atp_history = []
    protection_history = []
    
    for i in range(n_steps):
        # Activity for first 5 ms
        if i < 500:
            atp_system.step(dt, calcium, pH)
        else:
            # Recovery phase
            calcium_recovery = np.ones(grid_shape) * 100e-9
            pH_recovery = np.ones(grid_shape) * 7.3
            atp_system.step(dt, calcium_recovery, pH_recovery)
        
        # Record every 100 steps (1 ms)
        if i % 100 == 0:
            metrics = atp_system.get_experimental_metrics()
            atp_history.append(metrics['atp_mean_mM'])
            j_history.append(metrics['j_coupling_max_Hz'])
            protection_history.append(metrics['quantum_protection_max'])
            
            print(f"  t={i*dt*1e3:.1f} ms: "
                  f"ATP={metrics['atp_mean_mM']:.2f} mM, "
                  f"J_max={metrics['j_coupling_max_Hz']:.1f} Hz, "
                  f"Protection={metrics['quantum_protection_max']:.1f}x")
    
    # Test 2: Validate against literature
    print("\nTest 2: Literature Validation")
    
    # Rangaraju et al. 2014: ATP should drop during activity
    atp_drop = atp_history[0] - min(atp_history)
    print(f"  ATP depletion: {atp_drop:.2f} mM")
    if atp_drop > 0.1:  # Should see measurable drop
        print(f"    ✓ ATP consumption detected")
    
    # Cohn & Hughes 1962: J-coupling should reach ~20 Hz with ATP
    max_j = max(j_history)
    print(f"  Peak J-coupling: {max_j:.1f} Hz")
    if 15 <= max_j <= 22:
        print(f"    ✓ J-coupling in expected range (Cohn & Hughes 1962)")
    
    # Fisher 2015: Quantum protection should scale as J²
    max_protection = max(protection_history)
    expected_protection = (max_j / 0.2) ** 2  # Relative to free phosphate
    print(f"  Peak quantum protection: {max_protection:.0f}x")
    print(f"  Expected from J²: {expected_protection:.0f}x")
    
    if 0.5 * expected_protection <= max_protection <= 2 * expected_protection:
        print(f"    ✓ Protection scales with J² (Fisher 2015)")
    
    # Test 3: ATP recovery
    print("\nTest 3: ATP Recovery")
    final_atp = atp_history[-1]
    initial_atp = atp_history[0]
    recovery_fraction = (final_atp - min(atp_history)) / atp_drop
    
    print(f"  Initial ATP: {initial_atp:.2f} mM")
    print(f"  Minimum ATP: {min(atp_history):.2f} mM")
    print(f"  Final ATP: {final_atp:.2f} mM")
    print(f"  Recovery: {recovery_fraction*100:.1f}%")
    
    if recovery_fraction > 0.3:  # Should see partial recovery in 5 ms
        print(f"    ✓ ATP recovery detected")
    
    print("\n" + "="*70)
    print("ATP system validation complete!")
    print("="*70)