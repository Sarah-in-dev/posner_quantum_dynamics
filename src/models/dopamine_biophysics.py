"""
dopamine_biophysics.py

Biophysically constrained dopamine dynamics module for synaptic modeling.
Based on experimental literature values for vesicular release, diffusion, and uptake.

References:
- Pothos et al. 2000 J Neurosci: Vesicular content
- Garris et al. 1994 J Neurochem: Concentration ranges
- Rice & Cragg 2008 Brain Res Rev: Diffusion and spatial dynamics
- Cragg 2000 J Neurosci: DAT kinetics in primates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

@dataclass
class DopamineParameters:
    """
    Dopamine parameters from experimental literature.
    All values include citations for traceability.
    """
    
    # ============= VESICULAR CONTENT =============
    # Pothos et al. 2000 J Neurosci - cultured midbrain neurons
    molecules_per_vesicle: float = 3000  # molecules (range: 3000-7000)
    
    # Scientific Reports 2013 - striatal vesicles in vivo
    molecules_per_vesicle_striatum: float = 33000  # Much higher in vivo
    
    # Use lower estimate for conservative modeling
    quantal_size: float = 3000  # molecules per vesicle
    
    # ============= RELEASE PARAMETERS =============
    # Rice & Cragg 2008 Brain Res Rev
    vesicles_per_bouton: int = 20
    release_probability: float = 0.06  # per action potential
    n_release_sites: int = 2  # Simplified for model
    
    # ============= CONCENTRATIONS =============
    # Garris et al. 1994 J Neurochem
    tonic_concentration: float = 20e-9  # 20 nM baseline
    peak_concentration_synapse: float = 1.6e-6  # 1.6 µM peak
    
    # Colliver et al. 2000 J Neurosci
    vesicle_radius: float = 25e-9  # meters
    vesicle_concentration: float = 75e-3  # 75 mM inside vesicle
    
    # ============= DIFFUSION =============
    # Ford 2014 PNAS, Cragg & Rice 2004
    D_dopamine: float = 400e-12  # m²/s in extracellular space
    tortuosity: float = 1.6  # Striatal tortuosity factor
    D_effective: float = field(init=False)  # Computed
    
    # ============= DAT KINETICS =============
    # From voltammetry studies (see Cragg 2000, Ferris 2013)
    Km_DAT: float = 200e-9  # 200 nM (range: 100-1300 nM)
    Vmax_striatum: float = 3.0e-6  # 3 µM/s in dorsal striatum
    Vmax_NAc: float = 1.0e-6  # 1 µM/s in nucleus accumbens
    Vmax: float = 2.0e-6  # 2 µM/s (intermediate value)
    
    # ============= SPATIAL PARAMETERS =============
    # Rice & Cragg 2008 - sphere of influence
    sphere_of_influence: float = 7e-6  # 7 µm for D2 receptors
    release_site_spacing: float = 20e-6  # 20 µm between boutons
    release_spread_sigma: float = 1e-6  # 1 µm spread from release site
    
    # ============= TEMPORAL PARAMETERS =============
    burst_duration: float = 0.1  # 100 ms phasic burst
    burst_frequency: float = 20.0  # Hz during burst
    tonic_firing_rate: float = 4.0  # Hz tonic firing
    
    # ============= RECEPTOR BINDING =============
    # Dreyer et al. 2010, Richfield et al. 1989
    Kd_D1: float = 1e-9  # 1 nM (high affinity state)
    Kd_D2: float = 10e-9  # 10 nM
    
    # ============= MODULATION FACTORS =============
    # For integration with Model 5
    dopamine_enhances_dimer: float = 10.0  # Enhancement factor for dimer formation
    dopamine_suppresses_trimer: float = 0.1  # Suppression factor for trimer formation

    # ============= QUANTUM MODULATION PARAMETERS =============
    # Based on integration with Posner/dimer/trimer dynamics
    
    # Critical concentrations for quantum effects
    da_quantum_threshold: float = 100e-9  # 100 nM - D2 activation for quantum effects
    da_protection_threshold: float = 50e-9  # 50 nM - starts protecting coherence
    
    # Modulation strengths (dimensionless factors)
    dimer_enhancement_max: float = 10.0  # Max enhancement at saturating DA
    trimer_suppression_max: float = 0.1  # Max suppression (90% reduction)
    coherence_protection_max: float = 3.0  # Max T2 extension
    
    # Receptor-specific effects (based on Kd values)
    # D1: High affinity (1 nM) - might affect calcium dynamics
    # D2: Lower affinity (10 nM) - main quantum modulator
    d2_quantum_coupling: float = 10e-9  # Kd for quantum effects via D2
    
    # Temporal dynamics for quantum alignment
    da_calcium_delay: float = 0.050  # 50 ms optimal delay (Yagishita 2014)
    quantum_window: float = 0.200  # 200 ms window for coincidence
    
    def __post_init__(self):
        """Calculate derived parameters"""
        self.D_effective = self.D_dopamine / (self.tortuosity ** 2)
        print(f"Dopamine parameters initialized: D_eff={self.D_effective:.2e} m²/s")


class DopamineField:
    """
    Spatially resolved dopamine dynamics with biophysical constraints.
    
    This class handles:
    - Stochastic vesicular release
    - Spatial diffusion with tortuosity
    - Michaelis-Menten uptake via DAT
    - Receptor binding dynamics (optional)
    """
    
    def __init__(self, 
                 grid_size: int, 
                 dx: float,
                 params: Optional[DopamineParameters] = None,
                 release_sites: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize dopamine field.
        
        Args:
            grid_size: Number of grid points per dimension
            dx: Grid spacing in meters
            params: Dopamine parameters (uses defaults if None)
            release_sites: List of (i,j) tuples for release locations
        """
        self.params = params or DopamineParameters()
        self.grid_size = grid_size
        self.dx = dx
        
        # Initialize field to tonic level
        self.field = np.ones((grid_size, grid_size)) * self.params.tonic_concentration
        
        # Setup or use provided release sites
        if release_sites is not None:
            self.release_sites = release_sites
        else:
            self._setup_default_release_sites()
        
        # State tracking
        self.release_timer = 0.0
        self.vesicles_released = 0
        self.total_molecules_released = 0
        
        # Optional: track receptor occupancy
        self.D1_occupancy = np.zeros((grid_size, grid_size))
        self.D2_occupancy = np.zeros((grid_size, grid_size))
        
        logger.info(f"Dopamine field initialized: {grid_size}x{grid_size}, "
                   f"dx={dx*1e9:.1f}nm, {len(self.release_sites)} release sites")
    
    def _setup_default_release_sites(self):
        """Setup default release sites with biological spacing"""
        center = self.grid_size // 2
        spacing_grid = int(self.params.release_site_spacing / self.dx)
        
        # Two sites on either side of center
        self.release_sites = [
            (center - spacing_grid//2, center),
            (center + spacing_grid//2, center)
        ]
    
    def update(self, dt: float, stimulus: bool = False, reward: bool = False) -> Dict:
        """
        Update dopamine field for one timestep.
        
        Args:
            dt: Timestep in seconds
            stimulus: Whether there's synaptic stimulation
            reward: Whether there's reward signal (phasic burst)
            
        Returns:
            Dictionary of statistics
        """
        # 1. Handle release
        if reward:
            self._phasic_release(dt)
        elif stimulus:
            self._tonic_release(dt)
        else:
            self.release_timer = 0
        
        # 2. Diffusion
        self._apply_diffusion(dt)
        
        # 3. DAT uptake
        self._apply_uptake(dt)
        
        # 4. Update receptor occupancy (optional)
        self._update_receptor_occupancy()
        
        # 5. Enforce constraints
        self._enforce_constraints()
        
        return self.get_statistics()
    
    def _phasic_release(self, dt: float):
        """Phasic burst release during reward - more deterministic for learning"""
        if self.release_timer < self.params.burst_duration:
            # During learning, assume reliable burst firing
            # Release from all sites during burst
            for (i, j) in self.release_sites:
                # Simplified: during burst, release is more reliable
                # Instead of stochastic, use deterministic release during learning
                if self.release_timer < 0.1:  # First 100ms of burst
                    # Release multiple vesicles to achieve μM concentrations
                    vesicles_per_pulse = 5  # Multiple vesicles
                    self._release_vesicles(i, j, vesicles_per_pulse)
        
            self.release_timer += dt
    
    def _tonic_release(self, dt: float):
        """Tonic release during normal activity"""
        for (i, j) in self.release_sites:
            # Lower release probability
            p_release = self.params.release_probability * self.params.tonic_firing_rate * dt
            
            if np.random.random() < p_release:
                self._release_vesicles(i, j, 1)
    
    def _release_vesicles(self, i: int, j: int, n_vesicles: int):
        """Release vesicles at specified location with spatial spread"""
        # Calculate local concentration increase
        total_molecules = n_vesicles * self.params.quantal_size
        self.total_molecules_released += total_molecules
        self.vesicles_released += n_vesicles
        
        # Convert to concentration using actual voxel volume
        # Volume = dx * dx * cleft_width
        sigma = self.params.release_spread_sigma  # 100 nm
        cleft_width = 20e-9   # Add this line to define cleft_width
        
        # Volume of Gaussian spread in synaptic cleft
        # This is approximately the volume where 95% of the molecules will be
        effective_radius = 2 * sigma  # 2 sigma captures ~95%
        effective_area = np.pi * effective_radius**2
        effective_volume = effective_area * cleft_width

        avogadro = 6.022e23
        moles = total_molecules / avogadro
        concentration_increase = moles / effective_volume

        if self.vesicles_released <= 1:  # Print debug for first release
            print(f"DEBUG RELEASE: sigma={sigma*1e9:.1f}nm, effective_radius={effective_radius*1e9:.1f}nm")
            print(f"  Volume: {effective_volume:.2e} m³")
            print(f"  Concentration increase: {concentration_increase*1e9:.1f} nM")

        self._add_with_gaussian_spread(i, j, concentration_increase)

        if self.vesicles_released <= 1:
            print(f"  Max field after release: {np.max(self.field)*1e9:.1f} nM")
        
        # Apply spatial spread using Gaussian
        sigma_grid = self.params.release_spread_sigma / self.dx
    
    def _add_with_gaussian_spread(self, i: int, j: int, concentration: float):
        """Add concentration with Gaussian spatial spread"""
        sigma_grid = self.params.release_spread_sigma / self.dx
        
        # Create kernel
        kernel_size = min(int(6 * sigma_grid) + 1, self.grid_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate normalized Gaussian
        x = np.arange(kernel_size) - kernel_size // 2
        y = np.arange(kernel_size) - kernel_size // 2
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma_grid**2))
        kernel = kernel / np.sum(kernel)
        
        # Apply to field
        half = kernel_size // 2
        i_min = max(0, i - half)
        i_max = min(self.grid_size, i + half + 1)
        j_min = max(0, j - half)
        j_max = min(self.grid_size, j + half + 1)
        
        ki_min = half - (i - i_min)
        ki_max = half + (i_max - i)
        kj_min = half - (j - j_min)
        kj_max = half + (j_max - j)
        
        self.field[i_min:i_max, j_min:j_max] += \
            concentration * kernel[ki_min:ki_max, kj_min:kj_max]
    
    def _apply_diffusion(self, dt: float):
        """Apply diffusion with tortuosity"""
        # Laplacian with Neumann boundary conditions
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)
        
        laplacian = ndimage.convolve(self.field, kernel, mode='nearest')
        laplacian = laplacian / (self.dx * self.dx)
        
        self.field += self.params.D_effective * laplacian * dt
    
    def _apply_uptake(self, dt: float):
        """Apply Michaelis-Menten uptake via DAT"""
        # Only uptake above tonic level
        excess = np.maximum(0, self.field - self.params.tonic_concentration)
        
        # Michaelis-Menten kinetics
        uptake_rate = self.params.Vmax * excess / (self.params.Km_DAT + excess)
        self.field -= uptake_rate * dt
    
    def _update_receptor_occupancy(self):
        """Update D1 and D2 receptor occupancy (Hill equation)"""
        # Simple occupancy model
        self.D1_occupancy = self.field / (self.field + self.params.Kd_D1)
        self.D2_occupancy = self.field / (self.field + self.params.Kd_D2)
    
    def _enforce_constraints(self):
        """Enforce biological constraints"""
        # Minimum at tonic level
        self.field = np.maximum(self.field, self.params.tonic_concentration)
        
        # Maximum based on diluted vesicle concentration
        max_possible = 10e-6  # 10 µM is a reasonable maximum
        self.field = np.minimum(self.field, max_possible)
    
    def get_statistics(self) -> Dict:
        """Get current field statistics"""
        return {
            'min': np.min(self.field),
            'max': np.max(self.field),
            'mean': np.mean(self.field),
            'std': np.std(self.field),
            'release_site_values': [self.field[i,j] for i,j in self.release_sites],
            'vesicles_released': self.vesicles_released,
            'total_molecules': self.total_molecules_released,
            'D1_occupancy_mean': np.mean(self.D1_occupancy),
            'D2_occupancy_mean': np.mean(self.D2_occupancy)
        }
    
    def get_modulation_factor(self, i: int, j: int) -> Tuple[float, float]:
        """
        Get dopamine modulation factors for dimer/trimer formation.
        
        Returns:
            (dimer_factor, trimer_factor) at position (i,j)
        """
        da_level = self.field[i, j]
        
        # Sigmoid modulation based on D2 receptor activation
        d2_activation = da_level / (da_level + self.params.Kd_D2)
        
        # Enhance dimers, suppress trimers with dopamine
        dimer_factor = 1.0 + (self.params.dopamine_enhances_dimer - 1.0) * d2_activation
        trimer_factor = 1.0 - (1.0 - self.params.dopamine_suppresses_trimer) * d2_activation
        
        return dimer_factor, trimer_factor
    
    def reset(self):
        """Reset field to baseline"""
        self.field[:] = self.params.tonic_concentration
        self.release_timer = 0
        self.vesicles_released = 0
        self.total_molecules_released = 0

    def get_quantum_modulation(self, i: int, j: int) -> Dict[str, float]:
        """
        Calculate quantum modulation factors at a specific location.
        Returns factors for dimer enhancement, trimer suppression, and coherence protection.
        
        Based on:
        - Agarwal et al. 2023: Dimers superior for coherence
        - D2 receptor activation curve
        - Spatial gradients from release sites
        """
        da_local = self.field[i, j]
        
        # D2 receptor activation (Hill equation)
        d2_activation = da_local / (da_local + self.params.d2_quantum_coupling)
        
        # Calculate modulation factors
        factors = {
            'dimer_enhancement': 1.0 + (self.params.dimer_enhancement_max - 1.0) * d2_activation,
            'trimer_suppression': 1.0 - (1.0 - self.params.trimer_suppression_max) * d2_activation,
            'coherence_protection': 1.0 + (self.params.coherence_protection_max - 1.0) * d2_activation,
            'd2_occupancy': d2_activation,
            'above_quantum_threshold': da_local > self.params.da_quantum_threshold
        }
        
        return factors
    
    def calculate_spatiotemporal_gradient(self) -> np.ndarray:
        """
        Calculate spatial gradient of dopamine for directional effects.
        High gradients might indicate learning-relevant locations.
        """
        grad_x = np.gradient(self.field, axis=1) / self.dx
        grad_y = np.gradient(self.field, axis=0) / self.dx
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude
    
    def get_coincidence_factor(self, calcium_spike_time: float, 
                              current_time: float) -> float:
        """
        Calculate coincidence factor for STDP-like learning.
        Based on Yagishita et al. 2014 Science - dopamine must arrive
        within specific window after calcium.
        """
        time_diff = current_time - calcium_spike_time
        
        if 0 < time_diff < self.params.da_calcium_delay:
            # Too early - dopamine before optimal window
            return 0.5
        elif self.params.da_calcium_delay <= time_diff <= self.params.quantum_window:
            # Optimal window for quantum coherence
            return 1.0
        elif time_diff > self.params.quantum_window:
            # Too late - coherence already decayed
            return np.exp(-(time_diff - self.params.quantum_window) / 0.1)
        else:
            # Dopamine before calcium - no learning
            return 0.0
    
    def calculate_emergent_selectivity(self, i: int, j: int, 
                                    ca_local: float, po4_local: float) -> Dict[str, float]:
        """
        Calculate emergent dimer/trimer selectivity based on dopamine's 
        physical effects on the local environment.
    
        Key mechanisms:
        1. D2 activation reduces calcium influx
        2. Dopamine-phosphate complexation changes effective size
        3. Local ionic strength changes affect nucleation
        """
        da_local = self.field[i, j]
        d2_occupancy = da_local / (da_local + self.params.Kd_D2)
    
        # 1. D2 reduces calcium (via Gi/o → decreased Ca channel opening)
        # This changes the Ca/P ratio, affecting which phase forms
        ca_effective = ca_local * (1 - 0.3 * d2_occupancy)  # 30% reduction at full D2
    
        # 2. Calculate Ca/P ratio effects
        # Dimers need Ca6(PO4)4 → Ca/P = 1.5
        # Trimers need Ca9(PO4)6 → Ca/P = 1.5 (same ratio but different absolute amounts)
        ca_p_ratio = ca_effective / po4_local if po4_local > 0 else 0
    
        # With less calcium (D2 effect), smaller clusters are favored
        # because they need less total calcium to form
        if d2_occupancy > 0.5:  # Significant D2 activation

            # D2 doesn't reduce calcium for dimers (protected sites)
            ca_for_dimers = ca_local
            # But does reduce it for trimers (vulnerable sites)
            ca_for_trimers = ca_local * (1 - 0.3 * d2_occupancy)
            # Favor dimers: need only 6 Ca vs 9 Ca for trimers
            dimer_feasibility = min(ca_effective / (6 * 50e-9), 1.0)  # Can we make dimers?
            trimer_feasibility = min(ca_effective / (9 * 50e-9), 1.0)  # Can we make trimers?
        else:
            # Without D2, both equally feasible if enough substrate
            dimer_feasibility = 1.0
            trimer_feasibility = 1.0
    
        # 3. Dopamine affects local ionic strength via K+ channels
        # Higher ionic strength reduces Debye length → affects larger clusters more
        ionic_strength_factor = 1 + 0.5 * d2_occupancy
    
        # Trimers have more P-P interactions (15 vs 6), more affected by screening
        dimer_ionic_penalty = 1.0 / (1 + 0.1 * ionic_strength_factor)
        trimer_ionic_penalty = 1.0 / (1 + 0.3 * ionic_strength_factor)  # 3x more sensitive
    
        # 4. Combine effects (multiplicative because they're independent)
        dimer_factor = dimer_feasibility * dimer_ionic_penalty
        trimer_factor = trimer_feasibility * trimer_ionic_penalty
    
        return {
            'dimer_factor': dimer_factor,
            'trimer_factor': trimer_factor,
            'ca_effective': ca_effective,
            'd2_occupancy': d2_occupancy,
            'emergent_selectivity': dimer_factor / trimer_factor if trimer_factor > 0 else 10.0
        }

    # Convenience function for integration
def create_dopamine_system(grid_size: int, dx: float, 
                        custom_params: Optional[Dict] = None) -> DopamineField:
    """
    Create a dopamine system with optional custom parameters.
    
    Args:
        grid_size: Grid dimensions
        dx: Grid spacing in meters
        custom_params: Dictionary of parameter overrides
        
    Returns:
        Configured DopamineField instance
    """
    params = DopamineParameters()
    
    # Apply custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
    
    return DopamineField(grid_size, dx, params)