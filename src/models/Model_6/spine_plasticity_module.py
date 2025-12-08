"""
Spine Plasticity Module - Structural Information Storage
=========================================================

ARCHITECTURE: Follows Model 6 module pattern
- SpinePlasticityParameters: Literature-derived constants
- SpinePlasticityModule: Main class with step() and get_experimental_metrics()

INPUTS:
    - molecular_memory: CaMKII pT286 × GluN2B (from CaMKII module)
    - calcium_uM: Local calcium concentration
    
OUTPUTS:
    - spine_volume: Relative to baseline (1.0 = baseline)
    - AMPAR_count: Number of surface AMPARs
    - synaptic_strength: AMPAR / baseline (functional output)
    - actin_stable: Stable actin pool (consolidation marker)
    - phase: transient / stabilizing / consolidated

THE CENTRAL CLAIM:
    Spine morphology changes ARE the information storage.
    Volume increase = more AMPAR slots = stronger synapse.

LITERATURE SOURCES:
    - Matsuzaki et al. 2004 (Nat Neurosci): Spine enlargement kinetics
    - Bosch et al. 2014 (Neuron): Actin dynamics and phases
    - Makino & Bhalla 2018: Phase transitions in spine plasticity
    - Choquet & Bhalla 2025 (EMBO J): AMPAR trafficking review
    - Tang & bhalla 2017: Spine volume-strength relationship

Author: Model 6 Development
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS - ALL FROM LITERATURE
# =============================================================================

@dataclass
class ActinParameters:
    """
    Actin dynamics parameters from Bosch et al. 2014
    
    Two pools:
        - Dynamic: Rapidly turning over (branched F-actin at spine tip)
        - Stable: Slowly turning over (core structure, resists depolymerization)
    
    Transition: Dynamic → Stable via cofilin inhibition (LIMK pathway)
    """
    # Pool baselines (relative units, normalized so total = 1.0 at rest)
    dynamic_pool_baseline: float = 0.85  # 85% of actin is dynamic at rest (Honkura 2008)
    stable_pool_baseline: float = 0.15   # 15% is stable (Honkura 2008)
    
    # Polymerization (Arp2/3 mediated, calcium/CaMKII dependent)
    # Bosch et al. 2014: maximal polymerization ~2x baseline in first 2 min
    k_polymerization_max: float = 0.1    # s⁻¹, rate constant for new actin
    K_calcium_poly: float = 1.0          # μM, calcium EC50 for polymerization
    K_camkii_poly: float = 0.3           # molecular_memory EC50
    
    # Depolymerization (cofilin-mediated)
    k_depolymerization: float = 0.05     # s⁻¹ baseline depolymerization
    
    # Stabilization (LIMK → cofilin phosphorylation)
    # CaMKII activates LIMK which inhibits cofilin → stabilizes actin
    k_stabilization_max: float = 0.02    # s⁻¹, max rate dynamic→stable
    K_memory_stab: float = 0.5           # molecular_memory threshold
    
    # Destabilization (slow return to baseline)
    k_destabilization: float = 0.001     # s⁻¹, stable→dynamic (very slow)

    # Chemical Langevin noise (Bhalla lab modeling convention)
    # Noise amplitude scales as √(rate × dt) for molecular reactions
    stochastic: bool = True              # Enable Chemical Langevin noise

@dataclass  
class SpineVolumeParameters:
    """
    Spine volume dynamics from Matsuzaki et al. 2004, Makino & Bhalla 2018
    
    Key findings:
        - Maximum enlargement: ~390% for small spines, ~50% for large spines
        - Transient phase: τ ≈ 60s (rapid initial swelling)
        - Stabilization: requires sustained activity (>7 min)
        - Volume tracks actin content nearly linearly
    """
    # Volume limits
    baseline_volume: float = 1.0         # Normalized
    max_enlargement_ratio: float = 3.9   # 390% from Matsuzaki (small spines)
    min_volume_ratio: float = 0.5        # LTD can shrink spines
    
    # Volume-actin relationship
    # Bosch et al.: V ∝ (total actin content)^scaling
    actin_volume_scaling: float = 1.2    # Slightly supralinear
    
    # Phase thresholds (Makino & Bhalla 2018)
    stabilization_threshold: float = 2.0  # Volume > 2x baseline → stabilizing
    consolidation_time: float = 420.0     # 7 minutes for consolidation
    
    # Timescales
    tau_transient: float = 60.0          # seconds, Matsuzaki fast phase
    tau_volume_follow_actin: float = 5.0 # seconds, volume tracks actin

    # Thermal membrane fluctuations
    # From fluctuation-dissipation: σ_V/V ~ √(kT/κA) ~ 0.01-0.02 per √s
    stochastic: bool = True
    thermal_fluctuation_amplitude: float = 0.015  # Fractional noise per √second

@dataclass
class AMPARParameters:
    """
    AMPAR trafficking parameters from Bhalla 2014, Choquet & Bhalla 2025
    
    Three trafficking routes:
        1. Lateral diffusion (fast, τ ~ 5s)
        2. Exocytosis (slow, τ ~ 120s)  
        3. Endocytosis (removal, τ ~ 60s)
    
    Slot model: Spine volume determines available AMPAR slots
    """
    # Baseline counts
    baseline_AMPAR: float = 80.0         # Bhalla 2014: ~80 AMPARs per spine
    
    # Trafficking rates
    k_lateral_diffusion: float = 0.2     # s⁻¹ (τ ~ 5s), extrasynaptic→synaptic
    k_exocytosis: float = 0.008          # s⁻¹ (τ ~ 120s), intracellular→surface
    k_endocytosis: float = 0.017         # s⁻¹ (τ ~ 60s), removal
    
    # Slot model
    # Bhalla 2014: linear relationship V ∝ AMPAR_slots
    AMPAR_per_unit_volume: float = 80.0  # At baseline volume
    
    # Activity-dependent trafficking
    K_memory_exo: float = 0.3            # molecular_memory EC50 for exocytosis
    K_memory_lat: float = 0.2            # molecular_memory EC50 for lateral diff
    
    # Saturation
    max_AMPAR_ratio: float = 3.0         # Can't exceed 3x baseline

    # Stochastic trafficking (Bhalla 2014: individual receptor events)
    stochastic: bool = True              # Enable Poisson trafficking events
@dataclass
class SpinePlasticityParameters:
    """Combined parameters for spine plasticity module"""
    actin: ActinParameters = field(default_factory=ActinParameters)
    volume: SpineVolumeParameters = field(default_factory=SpineVolumeParameters)
    ampar: AMPARParameters = field(default_factory=AMPARParameters)
    
    def __post_init__(self):
        logger.info("Spine plasticity parameters initialized with literature values")


# =============================================================================
# MAIN MODULE
# =============================================================================

class SpinePlasticityModule:
    """
    Spine structural plasticity - the physical substrate of information storage
    
    This module tracks:
        1. Actin dynamics (the cytoskeleton that shapes the spine)
        2. Spine volume (the morphological change)
        3. AMPAR count (the functional readout)
    
    Architecture follows Model 6 pattern:
        - Initialize with parameters
        - step(dt, molecular_memory, calcium_uM) advances state
        - get_experimental_metrics() returns measurable outputs
    """
    
    def __init__(self, params: Optional[SpinePlasticityParameters] = None):
        """
        Initialize spine plasticity module
        
        Args:
            params: SpinePlasticityParameters (uses defaults if None)
        """
        self.params = params or SpinePlasticityParameters()

        # Use numpy's modern Generator for better statistical properties
        self.rng = np.random.default_rng()
        
        # State variables
        self._initialize_state()
        
        # Phase tracking
        self.phase = 'baseline'  # baseline → transient → stabilizing → consolidated
        self.time_above_threshold = 0.0
        
        # History for experimental analysis
        self.history = {
            'time': [],
            'actin_dynamic': [],
            'actin_stable': [],
            'actin_total': [],
            'spine_volume': [],
            'AMPAR_count': [],
            'synaptic_strength': [],
            'phase': []
        }
        
        self.time = 0.0
        
        logger.info("SpinePlasticityModule initialized")
        logger.info(f"  Baseline AMPAR: {self.params.ampar.baseline_AMPAR}")
        logger.info(f"  Max enlargement: {self.params.volume.max_enlargement_ratio}x")
        
    def _initialize_state(self):
        """Set initial state to baseline values"""
        p = self.params
        
        # Actin pools (relative units)
        self.actin_dynamic = p.actin.dynamic_pool_baseline
        self.actin_stable = p.actin.stable_pool_baseline
        
        # Derived quantities
        self.actin_total = self.actin_dynamic + self.actin_stable
        self.spine_volume = p.volume.baseline_volume
        self.AMPAR_count = p.ampar.baseline_AMPAR
        self.AMPAR_surface = p.ampar.baseline_AMPAR  # Surface pool
        self.AMPAR_reserve = p.ampar.baseline_AMPAR * 0.5  # Intracellular reserve
        
    def step(self, dt: float, molecular_memory: float, calcium_uM: float) -> Dict:
        """
        Advance spine plasticity by one timestep
        
        Args:
            dt: Timestep in seconds
            molecular_memory: CaMKII activation state (pT286 × GluN2B, range 0-1)
            calcium_uM: Local calcium concentration in μM
            
        Returns:
            Dict with current state metrics
        """
        self.time += dt
        
        # 1. Update actin dynamics
        self._update_actin(dt, molecular_memory, calcium_uM)
        
        # 2. Update spine volume (follows actin)
        self._update_volume(dt)
        
        # 3. Update AMPAR trafficking
        self._update_AMPAR(dt, molecular_memory)
        
        # 4. Update phase
        self._update_phase(dt)
        
        # 5. Record history
        self._record_history()
        
        return self.get_state()
        
    def _update_actin(self, dt: float, molecular_memory: float, calcium_uM: float):
        """
        Update actin pool dynamics with Chemical Langevin noise
        
        Based on Bosch et al. 2014:
            - High calcium/CaMKII → polymerization (new actin)
            - CaMKII → LIMK → cofilin inhibition → stabilization
            - Without activity → slow depolymerization/destabilization
        
        Stochastic: Chemical Langevin equation for molecular reaction noise
            σ = √(rate × dt) for each flux term
        """
        p = self.params.actin
        
        # Polymerization rate (calcium and CaMKII dependent)
        calcium_factor = calcium_uM**2 / (p.K_calcium_poly**2 + calcium_uM**2)
        memory_factor = molecular_memory / (p.K_camkii_poly + molecular_memory)
        k_poly = p.k_polymerization_max * calcium_factor * memory_factor
        
        # Depolymerization (baseline rate, reduced by memory)
        cofilin_activity = 1.0 / (1.0 + molecular_memory / 0.3)
        k_depoly = p.k_depolymerization * cofilin_activity
        
        # Stabilization rate (memory dependent)
        k_stab = p.k_stabilization_max * molecular_memory / (p.K_memory_stab + molecular_memory)
        
        # Destabilization (slow baseline return)
        k_destab = p.k_destabilization
        
        # Calculate fluxes (molecules per second, normalized units)
        flux_poly = k_poly
        flux_depoly = k_depoly * self.actin_dynamic
        flux_stab = k_stab * self.actin_dynamic
        flux_destab = k_destab * self.actin_stable
        
        # Deterministic changes
        d_dynamic = (flux_poly - flux_depoly - flux_stab + flux_destab) * dt
        d_stable = (flux_stab - flux_destab) * dt
        
        # Chemical Langevin noise: σ = √(flux × dt) for each reaction
        if p.stochastic:
            # Each flux contributes independent noise
            # Sign follows direction of flux contribution to each pool
            noise_poly = np.sqrt(abs(flux_poly) * dt) * self.rng.standard_normal()
            noise_depoly = np.sqrt(abs(flux_depoly) * dt) * self.rng.standard_normal()
            noise_stab = np.sqrt(abs(flux_stab) * dt) * self.rng.standard_normal()
            noise_destab = np.sqrt(abs(flux_destab) * dt) * self.rng.standard_normal()
            
            # Add noise to dynamic pool
            d_dynamic += noise_poly - noise_depoly - noise_stab + noise_destab
            
            # Add noise to stable pool (stabilization in, destabilization out)
            d_stable += noise_stab - noise_destab
        
        # Update with bounds
        self.actin_dynamic = max(0.1, self.actin_dynamic + d_dynamic)
        self.actin_stable = max(0.1, self.actin_stable + d_stable)
        self.actin_total = self.actin_dynamic + self.actin_stable
        
    def _update_volume(self, dt: float):
        """
        Update spine volume based on actin content with thermal fluctuations
        
        Matsuzaki et al. 2004: Volume tracks actin with τ ~ 5-60s
        
        Stochastic: Thermal membrane fluctuations from fluctuation-dissipation theorem
            σ_V/V ~ √(kT / κA) where κ is membrane bending modulus, A is area
            For dendritic spines: ~1-2% fluctuation amplitude per √second
        """
        p = self.params.volume
        
        # Target volume from actin content
        baseline_actin = p.baseline_volume
        target_volume = (self.actin_total / baseline_actin) ** p.actin_volume_scaling
        
        # Clamp to physiological limits
        target_volume = np.clip(target_volume, p.min_volume_ratio, p.max_enlargement_ratio)
        
        # Volume follows actin with timescale
        tau = p.tau_volume_follow_actin
        dV = (target_volume - self.spine_volume) / tau * dt
        
        # Thermal membrane fluctuations
        if p.stochastic:
            # Fluctuation-dissipation: noise amplitude ~ √(kT/κ) 
            # Empirically ~1.5% of volume per √second for spine membranes
            thermal_noise = (p.thermal_fluctuation_amplitude * self.spine_volume 
                            * np.sqrt(dt) * self.rng.standard_normal())
            dV += thermal_noise
        
        # Update with bounds
        self.spine_volume = np.clip(self.spine_volume + dV, 
                                    p.min_volume_ratio, p.max_enlargement_ratio)
        
    def _update_AMPAR(self, dt: float, molecular_memory: float):
        """
        Update AMPAR trafficking with Poisson stochastic events
        
        Three pathways:
            1. Lateral diffusion: extrasynaptic → synaptic (fast, τ ~ 5s)
            2. Exocytosis: intracellular → surface (slow, τ ~ 120s)
            3. Endocytosis: surface → intracellular (constitutive, τ ~ 60s)
        
        Stochastic: Individual receptor insertion/removal as Poisson events
            n_events ~ Poisson(λ) where λ = rate × pool_size × dt
        """
        p = self.params.ampar
        
        # Available slots (proportional to spine volume)
        max_slots = self.spine_volume * p.AMPAR_per_unit_volume * p.max_AMPAR_ratio
        available_slots = max(0, max_slots - self.AMPAR_surface)
        
        # Calculate rates
        k_lat = p.k_lateral_diffusion * (1.0 + molecular_memory / p.K_memory_lat)
        k_exo = p.k_exocytosis * (1.0 + 2.0 * molecular_memory / p.K_memory_exo)
        k_endo = p.k_endocytosis
        
        if p.stochastic:
            # Poisson events for discrete receptor trafficking
            # λ = rate × pool_size × dt (expected events per timestep)
            
            # Lateral diffusion: limited by available slots
            # Assume extrasynaptic pool is large, rate limited by slot availability
            lambda_lateral = k_lat * available_slots * 0.1 * dt
            n_lateral = self.rng.poisson(max(0, lambda_lateral))
            
            # Exocytosis: from reserve pool
            lambda_exo = k_exo * self.AMPAR_reserve * dt
            n_exo = self.rng.poisson(max(0, lambda_exo))
            
            # Endocytosis: from surface pool
            lambda_endo = k_endo * self.AMPAR_surface * dt
            n_endo = self.rng.poisson(max(0, lambda_endo))
            
            # Update surface pool (discrete changes)
            d_surface = float(n_lateral + n_exo - n_endo)
            
            # Update reserve pool
            d_reserve = float(-n_exo) + float(n_endo) * 0.5 + 0.01 * p.baseline_AMPAR * dt
            
        else:
            # Deterministic (original behavior)
            lateral_flux = k_lat * available_slots * 0.1
            exocytosis_flux = k_exo * self.AMPAR_reserve
            endocytosis_flux = k_endo * self.AMPAR_surface
            
            d_surface = (lateral_flux + exocytosis_flux - endocytosis_flux) * dt
            d_reserve = (-exocytosis_flux + endocytosis_flux * 0.5 + 0.01 * p.baseline_AMPAR) * dt
        
        # Apply updates with bounds
        self.AMPAR_surface = np.clip(self.AMPAR_surface + d_surface, 0.1, max_slots)
        self.AMPAR_reserve = max(0.1, self.AMPAR_reserve + d_reserve)
        
        # Total functional AMPAR count
        self.AMPAR_count = self.AMPAR_surface
        
    def _update_phase(self, dt: float):
        """
        Track plasticity phase transitions
        
        Phases (Makino & Bhalla 2018):
            - baseline: V < 1.2, no persistent activity
            - transient: V > 1.2, < 7 min since induction
            - stabilizing: V > 2.0, sustained > 7 min
            - consolidated: stable pool dominates, irreversible
        """
        p = self.params.volume
        
        # Track time above threshold
        if self.spine_volume > p.stabilization_threshold:
            self.time_above_threshold += dt
        else:
            # Reset if volume drops
            self.time_above_threshold = max(0, self.time_above_threshold - dt * 0.5)
        
        # Determine phase
        stable_fraction = self.actin_stable / max(0.1, self.actin_total)
        
        if self.spine_volume < 1.2:
            self.phase = 'baseline'
        elif self.time_above_threshold < p.consolidation_time:
            self.phase = 'transient'
        elif stable_fraction < 0.6:
            self.phase = 'stabilizing'
        else:
            self.phase = 'consolidated'
            
    def _record_history(self):
        """Record current state to history"""
        self.history['time'].append(self.time)
        self.history['actin_dynamic'].append(self.actin_dynamic)
        self.history['actin_stable'].append(self.actin_stable)
        self.history['actin_total'].append(self.actin_total)
        self.history['spine_volume'].append(self.spine_volume)
        self.history['AMPAR_count'].append(self.AMPAR_count)
        self.history['synaptic_strength'].append(self.get_synaptic_strength())
        self.history['phase'].append(self.phase)
        
    def get_state(self) -> Dict:
        """Get current state as dictionary"""
        return {
            'actin_dynamic': self.actin_dynamic,
            'actin_stable': self.actin_stable,
            'actin_total': self.actin_total,
            'spine_volume': self.spine_volume,
            'AMPAR_count': self.AMPAR_count,
            'AMPAR_surface': self.AMPAR_surface,
            'AMPAR_reserve': self.AMPAR_reserve,
            'synaptic_strength': self.get_synaptic_strength(),
            'phase': self.phase,
            'time_above_threshold': self.time_above_threshold
        }
        
    def get_synaptic_strength(self) -> float:
        """
        Calculate synaptic strength (the functional output)
        
        Defined as AMPAR_count / baseline
        This is what electrophysiology measures (EPSP amplitude)
        """
        return self.AMPAR_count / self.params.ampar.baseline_AMPAR
    
    def is_potentiated(self) -> bool:
        """Check if spine is in potentiated state (LTP)"""
        return self.spine_volume > 1.2 and self.get_synaptic_strength() > 1.2
    
    def get_experimental_metrics(self) -> Dict:
        """
        Get metrics that can be experimentally measured
        
        Returns:
            Dict with measurable quantities:
            - spine_volume_fold: Volume relative to baseline (2-photon microscopy)
            - synaptic_strength: EPSP amplitude / baseline (electrophysiology)
            - AMPAR_density: AMPARs per unit area (immunogold EM)
            - actin_stable_fraction: Stable / total actin (FRAP)
            - phase: Current plasticity phase
            - time_to_potentiation: Time until spine_volume > 1.5
        """
        metrics = {
            # Direct measurements
            'spine_volume_fold': self.spine_volume,
            'synaptic_strength': self.get_synaptic_strength(),
            'AMPAR_count': self.AMPAR_count,
            'AMPAR_density': self.AMPAR_count / self.spine_volume if self.spine_volume > 0 else 0,
            
            # Actin metrics (measurable by FRAP, phalloidin staining)
            'actin_total_fold': self.actin_total / (self.params.actin.dynamic_pool_baseline + 
                                                     self.params.actin.stable_pool_baseline),
            'actin_stable_fraction': self.actin_stable / max(0.1, self.actin_total),
            'actin_dynamic_fraction': self.actin_dynamic / max(0.1, self.actin_total),
            
            # Phase
            'phase': self.phase,
            'is_potentiated': self.is_potentiated(),
            
            # Kinetics (from history if available)
            'time_to_half_max_volume': self._find_time_to_threshold(
                self.history['spine_volume'], 
                0.5 * (self.params.volume.max_enlargement_ratio - 1) + 1
            ) if self.history['time'] else None,
            
            'time_to_potentiation': self._find_time_to_threshold(
                self.history['spine_volume'], 1.5
            ) if self.history['time'] else None,
        }
        
        # Add peak values from history
        if self.history['time']:
            metrics['peak_spine_volume'] = max(self.history['spine_volume'])
            metrics['peak_AMPAR_count'] = max(self.history['AMPAR_count'])
            metrics['peak_synaptic_strength'] = max(self.history['synaptic_strength'])
        
        return metrics
    
    def _find_time_to_threshold(self, values: list, threshold: float) -> Optional[float]:
        """Find first time when value exceeds threshold"""
        for i, v in enumerate(values):
            if v >= threshold:
                return self.history['time'][i]
        return None
    
    def reset(self):
        """Reset to baseline state"""
        self._initialize_state()
        self.phase = 'baseline'
        self.time_above_threshold = 0.0
        self.time = 0.0
        self.history = {k: [] for k in self.history}
        logger.info("SpinePlasticityModule reset to baseline")


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SPINE PLASTICITY MODULE - VALIDATION")
    print("="*70)
    
    # Initialize module
    module = SpinePlasticityModule()
    
    # Simulation parameters
    dt = 0.1  # 100 ms timestep
    duration = 600.0  # 10 minutes
    
    # Protocol: 60s of high activity, then monitor
    print("\nRunning protocol: 60s activity → 540s recovery")
    print("-"*70)
    
    t = 0.0
    while t < duration:
        # Activity pattern
        if t < 60.0:
            # High activity: elevated calcium and CaMKII
            calcium = 5.0  # μM
            molecular_memory = 0.8  # High CaMKII activation
        else:
            # Baseline: low calcium, decaying memory
            calcium = 0.1  # μM
            # Memory decays slowly
            molecular_memory = 0.8 * np.exp(-(t - 60.0) / 300.0)
        
        # Step
        state = module.step(dt, molecular_memory, calcium)
        
        # Print progress
        if t % 60.0 < dt:
            print(f"  t={t:5.0f}s: V={state['spine_volume']:.2f}, "
                  f"AMPAR={state['AMPAR_count']:.0f}, "
                  f"strength={state['synaptic_strength']:.2f}, "
                  f"phase={state['phase']}")
        
        t += dt
    
    # Get final metrics
    print("\n" + "="*70)
    print("EXPERIMENTAL METRICS")
    print("="*70)
    
    metrics = module.get_experimental_metrics()
    
    print(f"\nSpine morphology:")
    print(f"  Volume: {metrics['spine_volume_fold']:.2f}x baseline")
    print(f"  Peak volume: {metrics.get('peak_spine_volume', 0):.2f}x")
    print(f"  Time to 1.5x: {metrics['time_to_potentiation']:.1f}s" 
          if metrics['time_to_potentiation'] else "  Time to 1.5x: not reached")
    
    print(f"\nSynaptic strength:")
    print(f"  Current: {metrics['synaptic_strength']:.2f}x baseline")
    print(f"  Peak: {metrics.get('peak_synaptic_strength', 0):.2f}x")
    print(f"  AMPAR count: {metrics['AMPAR_count']:.0f}")
    
    print(f"\nActin dynamics:")
    print(f"  Total actin: {metrics['actin_total_fold']:.2f}x baseline")
    print(f"  Stable fraction: {metrics['actin_stable_fraction']:.1%}")
    print(f"  Dynamic fraction: {metrics['actin_dynamic_fraction']:.1%}")
    
    print(f"\nPhase: {metrics['phase']}")
    print(f"Potentiated: {metrics['is_potentiated']}")
    
    print("\n" + "="*70)
    print("✓ SPINE PLASTICITY MODULE VALIDATED")
    print("="*70)